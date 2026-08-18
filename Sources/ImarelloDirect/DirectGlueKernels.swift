import Foundation
import Metal

/// Bespoke elementwise/glue kernels for the direct-dispatch TE (Stage 2).
/// Heavy math (qmm, attention) comes from MLX's metallib; this file owns the
/// cheap ops so the whole layer can be encoded without MLX's runtime.
/// All accumulation in float32, matching the runtime's f32-RMSNorm convention.
enum DirectGlueKernels {
    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    // RMSNorm: one threadgroup per row, f32 accumulate.
    // y = x * rsqrt(mean(x^2) + eps) * w
    kernel void dq_rmsnorm(
        device const half* x [[buffer(0)]],
        device const float* w [[buffer(1)]],
        device half* y [[buffer(2)]],
        constant int& d [[buffer(3)]],
        constant float& eps [[buffer(4)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint tpsz [[threads_per_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sgid [[simdgroup_index_in_threadgroup]]) {
        device const half* xr = x + (size_t)row * d;
        float acc = 0.0f;
        for (int i = tid; i < d; i += tpsz) {
            float v = (float)xr[i];
            acc = fma(v, v, acc);
        }
        acc = simd_sum(acc);
        threadgroup float parts[32];
        threadgroup float invShared;
        if (lane == 0) parts[sgid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            uint ngroups = (tpsz + 31) / 32;
            for (uint i = 0; i < ngroups; i++) total += parts[i];
            invShared = rsqrt(total / (float)d + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = invShared;
        device half* yr = y + (size_t)row * d;
        for (int i = tid; i < d; i += tpsz) {
            yr[i] = (half)((float)xr[i] * inv * w[i]);
        }
    }

    // Non-traditional (half-split) RoPE over [L, H, D] rows, position = L index.
    // Pairs (i, i + D/2): out_i = x_i*cos - x_j*sin ; out_j = x_j*cos + x_i*sin
    // theta_i = pos * base^(-2i/D)
    kernel void dq_rope(
        device const half* x [[buffer(0)]],
        device half* y [[buffer(1)]],
        constant int& H [[buffer(2)]],
        constant int& D [[buffer(3)]],
        constant float& base [[buffer(4)]],
        uint3 gid [[thread_position_in_grid]]) {
        int i = gid.x;
        int h = gid.y;
        int l = gid.z;
        int halfD = D / 2;
        if (i >= halfD) return;
        size_t off = ((size_t)l * H + h) * D;
        float theta = (float)l * metal::exp2(-2.0f * (float)i / (float)D * metal::log2((float)base));
        float c = metal::cos(theta);
        float s = metal::sin(theta);
        float a = (float)x[off + i];
        float b = (float)x[off + i + halfD];
        y[off + i] = (half)(a * c - b * s);
        y[off + i + halfD] = (half)(b * c + a * s);
    }

    // y = silu(g) * u = (g * sigmoid(g)) * u
    kernel void dq_silu_mul(
        device const half* g [[buffer(0)]],
        device const half* u [[buffer(1)]],
        device half* y [[buffer(2)]],
        constant uint& n [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
        if (gid >= n) return;
        float gv = (float)g[gid];
        float s = 1.0f / (1.0f + metal::exp(-gv));
        y[gid] = (half)(gv * s * (float)u[gid]);
    }

    // y = a + b
    kernel void dq_add(
        device const half* a [[buffer(0)]],
        device const half* b [[buffer(1)]],
        device half* y [[buffer(2)]],
        constant uint& n [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
        if (gid >= n) return;
        y[gid] = (half)((float)a[gid] + (float)b[gid]);
    }
    """

    static func makeLibrary(device: MTLDevice) throws -> MTLLibrary {
        let options = MTLCompileOptions()
        options.mathMode = .fast
        return try device.makeLibrary(source: source, options: options)
    }
}
