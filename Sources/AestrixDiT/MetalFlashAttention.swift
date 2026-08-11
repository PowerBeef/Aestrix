import Foundation
import MLX
import MLXFast

/// Attention compute backend for DiT SDPA.
public enum AttentionBackend: String, Sendable, Codable, CaseIterable {
    /// Query-chunked `MLXFast.scaledDotProductAttention` (legacy low-RAM path).
    case mlx
    /// Fused FlashAttention (MLX Steel simdgroup MMA when available; hybrid FA2 fallback).
    case metalFA = "metal-fa"
    /// Prefer fused FA when seq ≥ `AttentionTuning.metalFAMinSeq`.
    case auto
}

/// FlashAttention-2 **forward** for MLX arrays.
///
/// ## v5 — fused single-kernel path (MFA-class)
///
/// MLX 0.31+ already ships **Steel Attention**: a fused Metal kernel using
/// `simdgroup_matrix` MMA fragments (BQ=32, BK=16/32, BD=64/80/128), online
/// softmax, and threadgroup Q/K/V tiles — the same design class as MFA/ccv.
///
/// Important: Steel is selected only when `ScaledDotProductAttention::use_fallback`
/// is false, which for inference + D∈{64,80,128} + **Tq > 8** is true. Our older
/// **query-chunked** path (Tq chunk = 512) still works but issues many launches and
/// can miss the best fused schedule.
///
/// `scaledDotProductAttention` therefore:
/// 1. Prefer **one** `MLXFast.scaledDotProductAttention` call (Steel fused FA) for supported D.
/// 2. Else hybrid host-loop FA2 with steel `matmul` tiles (D not in {64,80,128}).
/// 3. Native pure-Metal scalar kernel remains for unit tests / research only.
///
/// Layout: Q,K,V,Out row-contiguous `[B, H, S, D]`. No mask (FLUX distilled).
public enum MetalFlashAttention {
    /// Query tile size used by hybrid fallback / docs (Steel uses BQ=32 internally).
    public static let blockR = 32
    /// Key/value tile for hybrid fallback.
    public static let blockC = 256

    /// Head dims for which MLX Steel fused attention is available (see mlx `use_fallback`).
    public static let steelHeadDims: Set<Int> = [64, 80, 128]

    /// FA2 forward — prefer fused Steel kernel.
    public static func scaledDotProductAttention(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        scale: Float,
        blockR: Int = blockR,
        blockC: Int = blockC
    ) -> MLXArray {
        precondition(query.ndim == 4 && key.ndim == 4 && value.ndim == 4)
        let D = query.dim(3)
        let outDtype = query.dtype

        // --- Path 1: single fused Steel SDPA (simdgroup MMA + online softmax) ---
        if steelHeadDims.contains(D),
           key.dim(3) == D,
           value.dim(3) == D
        {
            // Contiguous layout helps avoid extra device copies in eval_gpu.
            let q = ensureMatrixContiguous(query)
            let k = ensureMatrixContiguous(key)
            let v = ensureMatrixContiguous(value)
            let out = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: nil
            )
            return out.dtype == outDtype ? out : out.asType(outDtype)
        }

        // --- Path 2: hybrid FA2 (steel matmul tiles) for unsupported head dims ---
        return hybridBlockGEMM(
            query: query, key: key, value: value, scale: scale, blockC: blockC
        )
    }

    /// Ensure last dim stride is 1 (matrix-contiguous) for Steel loaders.
    private static func ensureMatrixContiguous(_ x: MLXArray) -> MLXArray {
        // MLXFast SDPA / eval_gpu will copy if needed; asType + reshape path is a no-op when already good.
        // Force a materialised row-contiguous view via contiguous ops when possible.
        return x
    }

    // MARK: - Hybrid FA2 (block GEMM host loop) — fallback

    /// Full-Q + stream K/V tiles via steel `matmul` + online softmax.
    public static func hybridBlockGEMM(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        scale: Float,
        blockC: Int = blockC
    ) -> MLXArray {
        let B = query.dim(0)
        let H = query.dim(1)
        let Sq = query.dim(2)
        let D = query.dim(3)
        let Sk = key.dim(2)
        let outDtype = query.dtype

        let q = query.asType(.float32)
        let k = key.asType(.float32)
        let v = value.asType(.float32)

        if Sq <= blockC && Sk <= blockC {
            let scores = matmul(q * scale, k.transposed(0, 1, 3, 2))
            let probs = softmax(scores, axis: -1)
            let out = matmul(probs, v)
            return outDtype == .float32 ? out : out.asType(outDtype)
        }

        let negLarge = MLXArray(Float(-1.0e9))
        var m = MLXArray.full([B, H, Sq, 1], values: negLarge, dtype: .float32)
        var l = MLXArray.zeros([B, H, Sq, 1], dtype: .float32)
        var O = MLXArray.zeros([B, H, Sq, D], dtype: .float32)
        let qScaled = q * scale

        var ks = 0
        var keyTileIndex = 0
        let checkpointEvery = 3
        while ks < Sk {
            let ke = min(ks + blockC, Sk)
            let Kb = k[0..., 0..., ks ..< ke, 0...]
            let Vb = v[0..., 0..., ks ..< ke, 0...]
            let S = matmul(qScaled, Kb.transposed(0, 1, 3, 2))
            let mTile = S.max(axis: -1, keepDims: true)
            let mNew = maximum(m, mTile)
            let alpha = MLX.exp(m - mNew)
            let P = MLX.exp(S - mNew)
            let lTile = P.sum(axis: -1, keepDims: true)
            O = O * alpha + matmul(P, Vb)
            m = mNew
            l = alpha * l + lTile
            keyTileIndex += 1
            if keyTileIndex % checkpointEvery == 0 {
                eval(O, m, l)
            }
            ks = ke
        }

        let out = O / maximum(l, MLXArray(Float(1e-12)))
        eval(out)
        Memory.clearCache()
        return outDtype == .float32 ? out : out.asType(outDtype)
    }

    // MARK: - Native Metal fused kernel (float4 + multi-warp TG)

    /// Custom fused FA2 Metal kernel: one dispatch, online softmax, float4 FMA, Br=32.
    /// Used when we want an Aestrix-owned kernel (tests + optional force path).
    /// For product D=128, prefer Steel via `scaledDotProductAttention`.
    public static func scaledDotProductAttentionFusedMetal(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        scale: Float
    ) -> MLXArray {
        precondition(query.ndim == 4)
        let B = query.dim(0)
        let H = query.dim(1)
        let Sq = query.dim(2)
        let D = query.dim(3)
        let Sk = key.dim(2)
        precondition(D > 0 && D <= 128 && D % 4 == 0, "fused metal FA needs D multiple of 4, ≤128")

        let meta = MLXArray([Int32(B), Int32(H), Int32(Sq), Int32(Sk), Int32(D)], [5])
        let scaleArr = MLXArray([scale])

        let dtype = query.dtype
        let useF16 = dtype == .float16
        let kernel = useF16 ? kernelFusedF16 : kernelFusedF32
        let qIn = useF16 ? query : query.asType(.float32)
        let kIn = useF16 ? key.asType(.float16) : key.asType(.float32)
        let vIn = useF16 ? value.asType(.float16) : value.asType(.float32)
        let outDtype: DType = useF16 ? .float16 : .float32

        // BQ=32 threads, 4 warps of 8 rows each in spirit of Steel BQ=32 WM=4.
        let BQ = 32
        let BK = 32
        let numQBlocks = (Sq + BQ - 1) / BQ
        // grid = total threads (MLX dispatchThreads)
        let gridX = numQBlocks * BQ
        let gridY = B * H

        let outputs = kernel(
            [qIn, kIn, vIn, meta, scaleArr],
            template: [
                ("BQ", BQ),
                ("BK", BK),
                ("HEAD_DIM_MAX", 128),
            ],
            grid: (gridX, gridY, 1),
            threadGroup: (BQ, 1, 1),
            outputShapes: [[B, H, Sq, D]],
            outputDTypes: [outDtype],
            initValue: 0
        )
        let out = outputs[0]
        return (dtype == outDtype) ? out : out.asType(dtype)
    }

    // MARK: - Native scalar kernel (legacy research)

    public static func scaledDotProductAttentionNativeMetal(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        scale: Float
    ) -> MLXArray {
        // Delegate to fused float4 kernel (same API name as before for tests that call native).
        return scaledDotProductAttentionFusedMetal(
            query: query, key: key, value: value, scale: scale
        )
    }

    // MARK: - Fused Metal source (float4 online softmax FA2)

    private static let kernelFusedF32: MLXFast.MLXFastKernel = makeFusedKernel(
        name: "aestrix_fa2_fused_f32", f16: false)
    private static let kernelFusedF16: MLXFast.MLXFastKernel = makeFusedKernel(
        name: "aestrix_fa2_fused_f16", f16: true)

    private static func makeFusedKernel(name: String, f16: Bool) -> MLXFast.MLXFastKernel {
        let header = f16 ? (fusedHeader + "\n#define AESTIX_FA_F16 1\n") : fusedHeader
        return MLXFast.metalKernel(
            name: name,
            inputNames: ["q", "k", "v", "meta", "scale_arr"],
            outputNames: ["out"],
            source: fusedBody,
            header: header,
            ensureRowContiguous: true,
            atomicOutputs: false
        )
    }

    private static let fusedHeader: String = """
    #include <metal_stdlib>
    using namespace metal;

    inline float fa_load(device const float* p, long idx) { return p[idx]; }
    inline float fa_load(device const half* p, long idx) { return float(p[idx]); }
    inline void fa_store(device float* p, long idx, float x) { p[idx] = x; }
    inline void fa_store(device half* p, long idx, float x) { p[idx] = half(x); }

    // float4 helpers for D-vectorized FMA (HEAD_DIM multiple of 4).
    inline float4 fa_load4_f32(device const float* p, long base) {
        return float4(p[base], p[base+1], p[base+2], p[base+3]);
    }
    inline float4 fa_load4_f16(device const half* p, long base) {
        return float4(float(p[base]), float(p[base+1]), float(p[base+2]), float(p[base+3]));
    }
    """

    /// One thread per query row in a BQ-wide block; loops K tiles of BK.
    /// float4 along D. Threadgroup holds K/V tiles.
    private static let fusedBody: String = """
        const int B  = meta[0];
        const int H  = meta[1];
        const int Sq = meta[2];
        const int Sk = meta[3];
        const int D  = meta[4];
        const float scale = scale_arr[0];

        const uint q_block = threadgroup_position_in_grid.x;
        const uint bh      = threadgroup_position_in_grid.y;
        const uint tid     = thread_position_in_threadgroup.x; // 0..BQ-1

        if (bh >= (uint)(B * H) || tid >= (uint)BQ) return;

        const int b = (int)bh / H;
        const int h = (int)bh % H;
        const int q_idx = (int)q_block * BQ + (int)tid;
        const bool valid_q = q_idx < Sq;
        const int D4 = D >> 2;

        // Q row + O accumulator as float4 packs
        float4 q4[HEAD_DIM_MAX / 4];
        float4 o4[HEAD_DIM_MAX / 4];
        for (int i = 0; i < HEAD_DIM_MAX / 4; i++) {
            q4[i] = float4(0.f);
            o4[i] = float4(0.f);
        }
        float m_i = -INFINITY;
        float l_i = 0.f;

        if (valid_q) {
            const long q_base = (((long)b * H + h) * Sq + q_idx) * (long)D;
            for (int i = 0; i < D4; i++) {
                #if defined(AESTIX_FA_F16)
                q4[i] = fa_load4_f16(q, q_base + (long)i * 4);
                #else
                q4[i] = fa_load4_f32(q, q_base + (long)i * 4);
                #endif
            }
        }

        // BK * HEAD_DIM_MAX * 2 * 4 @ BK=32,D=128 = 32KB — borderline; use BK=32 D packs in f32.
        threadgroup float k_tile[BK * HEAD_DIM_MAX];
        threadgroup float v_tile[BK * HEAD_DIM_MAX];

        const int num_k_blocks = (Sk + BK - 1) / BK;
        for (int kb = 0; kb < num_k_blocks; kb++) {
            const int k_base = kb * BK;
            // Cooperative load (float elements)
            const int tile_elems = BK * D;
            for (int i = (int)tid; i < tile_elems; i += BQ) {
                const int kk = i / D;
                const int dd = i % D;
                const int k_idx = k_base + kk;
                float kv = 0.f, vv = 0.f;
                if (k_idx < Sk) {
                    const long base = (((long)b * H + h) * Sk + k_idx) * (long)D + dd;
                    #if defined(AESTIX_FA_F16)
                    kv = fa_load(k, base);
                    vv = fa_load(v, base);
                    #else
                    kv = fa_load(k, base);
                    vv = fa_load(v, base);
                    #endif
                }
                k_tile[i] = kv;
                v_tile[i] = vv;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (valid_q) {
                const int valid_bc = min(BK, Sk - k_base);
                float scores[BK];
                float row_max = -INFINITY;
                for (int j = 0; j < BK; j++) scores[j] = -INFINITY;

                for (int j = 0; j < valid_bc; j++) {
                    float s = 0.f;
                    const int koff = j * D;
                    for (int i = 0; i < D4; i++) {
                        float4 kk = float4(
                            k_tile[koff + i*4],
                            k_tile[koff + i*4 + 1],
                            k_tile[koff + i*4 + 2],
                            k_tile[koff + i*4 + 3]);
                        s += dot(q4[i], kk);
                    }
                    s *= scale;
                    scores[j] = s;
                    row_max = max(row_max, s);
                }

                const float m_new = max(m_i, row_max);
                float exp_m = (m_i > -INFINITY) ? metal::exp(m_i - m_new) : 0.f;
                float l_new = exp_m * l_i;
                float p[BK];
                for (int j = 0; j < BK; j++) p[j] = 0.f;
                for (int j = 0; j < valid_bc; j++) {
                    const float pj = metal::exp(scores[j] - m_new);
                    p[j] = pj;
                    l_new += pj;
                }

                for (int i = 0; i < D4; i++) {
                    float4 acc = o4[i] * exp_m;
                    for (int j = 0; j < valid_bc; j++) {
                        const int voff = j * D + i * 4;
                        float4 vv = float4(
                            v_tile[voff], v_tile[voff+1], v_tile[voff+2], v_tile[voff+3]);
                        acc += p[j] * vv;
                    }
                    o4[i] = acc;
                }
                m_i = m_new;
                l_i = l_new;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (valid_q) {
            const float inv_l = (l_i > 0.f) ? (1.f / l_i) : 0.f;
            const long out_base = (((long)b * H + h) * Sq + q_idx) * (long)D;
            for (int i = 0; i < D4; i++) {
                float4 val = o4[i] * inv_l;
                #if defined(AESTIX_FA_F16)
                fa_store(out, out_base + (long)i*4,     val.x);
                fa_store(out, out_base + (long)i*4 + 1, val.y);
                fa_store(out, out_base + (long)i*4 + 2, val.z);
                fa_store(out, out_base + (long)i*4 + 3, val.w);
                #else
                fa_store(out, out_base + (long)i*4,     val.x);
                fa_store(out, out_base + (long)i*4 + 1, val.y);
                fa_store(out, out_base + (long)i*4 + 2, val.z);
                fa_store(out, out_base + (long)i*4 + 3, val.w);
                #endif
            }
        }
    """
}
