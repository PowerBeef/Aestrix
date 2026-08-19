import Foundation
import Metal

/// Fused glue kernels for the direct DiT single-stream block (Stage 2, F-track).
/// The ÷16/×16 f16-qmm scale protocol folds in as follows: it CANCELS inside
/// the q/k RMSNorm (rms is scale-invariant), V and the SwiGLU input carry ×16,
/// and the whole `to_out` input carries ÷16 — so the direct block reproduces
/// the product math with zero standalone rescale passes except one in-place
/// scale over the attention half of the concat buffer.
enum DirectDiTKernels {
    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    // LayerNorm (no affine, eps 1e-6, f32) + AdaLN modulate + ÷16 → f16.
    kernel void dd_ln_mod_prescale(
        device const float* x [[buffer(0)]],
        device const float* scaleV [[buffer(1)]],
        device const float* shiftV [[buffer(2)]],
        device half* y [[buffer(3)]],
        constant int& d [[buffer(4)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint tpsz [[threads_per_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sgid [[simdgroup_index_in_threadgroup]]) {
        device const float* xr = x + (size_t)row * d;
        float sum = 0.0f, sq = 0.0f;
        for (int i = tid; i < d; i += tpsz) {
            float v = xr[i];
            sum += v;
            sq = fma(v, v, sq);
        }
        sum = simd_sum(sum);
        sq = simd_sum(sq);
        threadgroup float pSum[32], pSq[32];
        threadgroup float meanSh, invSh;
        if (lane == 0) { pSum[sgid] = sum; pSq[sgid] = sq; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float ts = 0.0f, tq = 0.0f;
            uint ng = (tpsz + 31) / 32;
            for (uint i = 0; i < ng; i++) { ts += pSum[i]; tq += pSq[i]; }
            float mean = ts / (float)d;
            float var = tq / (float)d - mean * mean;
            meanSh = mean;
            invSh = rsqrt(var + 1e-6f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mean = meanSh, inv = invSh;
        device half* yr = y + (size_t)row * d;
        for (int i = tid; i < d; i += tpsz) {
            float n = (xr[i] - mean) * inv;
            float m = fma(scaleV[i] + 1.0f, n, shiftV[i]);
            yr[i] = (half)(m * 0.0625f);
        }
    }

    // RMSNorm over pitched head rows of the fused proj (f16 in, f32 math,
    // f16 out, CONTIGUOUS [L, H, D]). The qmm ×16 post-scale cancels here.
    kernel void dd_rmsnorm_pitched(
        device const half* x [[buffer(0)]],
        device const float* w [[buffer(1)]],
        device half* y [[buffer(2)]],
        constant int& d [[buffer(3)]],
        constant int& rowPitch [[buffer(4)]],
        constant int& sectionOff [[buffer(5)]],
        constant int& headsPerRow [[buffer(6)]],
        constant float& eps [[buffer(7)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint tpsz [[threads_per_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sgid [[simdgroup_index_in_threadgroup]]) {
        int l = row / headsPerRow;
        int h = row % headsPerRow;
        device const half* xr = x + (size_t)l * rowPitch + sectionOff + h * d;
        float acc = 0.0f;
        for (int i = tid; i < d; i += tpsz) {
            float v = (float)xr[i];
            acc = fma(v, v, acc);
        }
        acc = simd_sum(acc);
        threadgroup float parts[32];
        threadgroup float invSh;
        if (lane == 0) parts[sgid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float t = 0.0f;
            uint ng = (tpsz + 31) / 32;
            for (uint i = 0; i < ng; i++) t += parts[i];
            invSh = rsqrt(t / (float)d + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = invSh;
        device half* yr = y + (size_t)row * d;
        for (int i = tid; i < d; i += tpsz) {
            yr[i] = (half)((float)xr[i] * inv * w[i]);
        }
    }

    // FLUX interleaved RoPE on contiguous [L, H, D] f16; cos/sin [L, D/2] f32.
    kernel void dd_rope_interleaved(
        device const half* x [[buffer(0)]],
        device half* y [[buffer(1)]],
        device const float* cosT [[buffer(2)]],
        device const float* sinT [[buffer(3)]],
        constant int& H [[buffer(4)]],
        constant int& D [[buffer(5)]],
        uint3 gid [[thread_position_in_grid]]) {
        int i = gid.x, h = gid.y, l = gid.z;
        int halfD = D / 2;
        if (i >= halfD) return;
        size_t off = ((size_t)l * H + h) * D + 2 * i;
        float c = cosT[l * halfD + i];
        float s = sinT[l * halfD + i];
        float re = (float)x[off];
        float im = (float)x[off + 1];
        y[off] = (half)(re * c - im * s);
        y[off + 1] = (half)(im * c + re * s);
    }

    // Pitched section copy with scale (V extraction: ×16 restores the qmm scale).
    kernel void dd_scale_cast_pitched(
        device const half* x [[buffer(0)]],
        device half* y [[buffer(1)]],
        constant int& rowPitch [[buffer(2)]],
        constant int& sectionOff [[buffer(3)]],
        constant int& width [[buffer(4)]],
        constant float& scale [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]) {
        int j = gid.x, l = gid.y;
        if (j >= width) return;
        y[(size_t)l * width + j] =
            (half)((float)x[(size_t)l * rowPitch + sectionOff + j] * scale);
    }

    // SwiGLU over pitched g/u proj sections, writing into the concat buffer.
    // inScale restores the qmm ×16; outScale carries the to_out ÷16.
    kernel void dd_swiglu_pitched(
        device const half* x [[buffer(0)]],
        device half* y [[buffer(1)]],
        constant int& rowPitch [[buffer(2)]],
        constant int& gOff [[buffer(3)]],
        constant int& uOff [[buffer(4)]],
        constant int& width [[buffer(5)]],
        constant int& outPitch [[buffer(6)]],
        constant int& outOff [[buffer(7)]],
        constant float& inScale [[buffer(8)]],
        constant float& outScale [[buffer(9)]],
        uint2 gid [[thread_position_in_grid]]) {
        int j = gid.x, l = gid.y;
        if (j >= width) return;
        size_t base = (size_t)l * rowPitch;
        float g = (float)x[base + gOff + j] * inScale;
        float u = (float)x[base + uOff + j] * inScale;
        float sig = 1.0f / (1.0f + metal::exp(-g));
        y[(size_t)l * outPitch + outOff + j] = (half)(g * sig * u * outScale);
    }

    // In-place scale over a pitched column range (attention half of concat ÷16).
    kernel void dd_scale_inplace(
        device half* x [[buffer(0)]],
        constant int& rowPitch [[buffer(1)]],
        constant int& off [[buffer(2)]],
        constant int& width [[buffer(3)]],
        constant float& scale [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]) {
        int j = gid.x, l = gid.y;
        if (j >= width) return;
        size_t idx = (size_t)l * rowPitch + off + j;
        x[idx] = (half)((float)x[idx] * scale);
    }

    // f32 → f16 with scale (latents → qmm input, ÷16).
    kernel void dd_cast_prescale(
        device const float* x [[buffer(0)]],
        device half* y [[buffer(1)]],
        constant uint& n [[buffer(2)]],
        constant float& s [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
        if (gid >= n) return;
        y[gid] = (half)(x[gid] * s);
    }

    // f16 → f32 with scale (qmm out ×16 → residual stream).
    kernel void dd_cast_postscale(
        device const half* x [[buffer(0)]],
        device float* y [[buffer(1)]],
        constant uint& n [[buffer(2)]],
        constant float& s [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
        if (gid >= n) return;
        y[gid] = (float)x[gid] * s;
    }

    // Gated residual: y = x + gate ⊙ (v · scale); x f32, v f16 (qmm out), y f32.
    kernel void dd_gate_add(
        device const float* x [[buffer(0)]],
        device const float* gateV [[buffer(1)]],
        device const half* v [[buffer(2)]],
        device float* y [[buffer(3)]],
        constant int& d [[buffer(4)]],
        constant float& scale [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]) {
        int j = gid.x, l = gid.y;
        if (j >= d) return;
        size_t idx = (size_t)l * d + j;
        y[idx] = fma(gateV[j], (float)v[idx] * scale, x[idx]);
    }
    """

    static func makeLibrary(device: MTLDevice) throws -> MTLLibrary {
        let options = MTLCompileOptions()
        options.mathMode = .fast
        return try device.makeLibrary(source: source, options: options)
    }
}
