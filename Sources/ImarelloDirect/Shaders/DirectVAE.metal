#include <metal_stdlib>
using namespace metal;

// GroupNorm, three-pass coalesced. Pass 1: per-(channel, chunk) partial
// sums — consecutive threads read consecutive channels, so warp reads are
// contiguous. partial layout: [(chunk*C + c)*2 + {sum, sumsq}].
kernel void dv_gn_partial(
    device const float* x [[buffer(0)]],
    device float* partial [[buffer(1)]],
    constant int& HW [[buffer(2)]],
    constant int& C [[buffer(3)]],
    constant int& chunks [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    int c = gid.x, chunk = gid.y;
    if (c >= C || chunk >= chunks) return;
    int rows = (HW + chunks - 1) / chunks;
    int r0 = chunk * rows;
    int r1 = min(HW, r0 + rows);
    float sum = 0.0f, sq = 0.0f;
    for (int r = r0; r < r1; r++) {
        float v = x[(size_t)r * C + c];
        sum += v;
        sq = fma(v, v, sq);
    }
    size_t idx = ((size_t)chunk * C + c) * 2;
    partial[idx] = sum;
    partial[idx + 1] = sq;
}

// Pass 2: reduce a group's partials to (mean, inv_std). One TG per group.
kernel void dv_gn_finalize(
    device const float* partial [[buffer(0)]],
    device float* stats [[buffer(1)]],
    constant int& HW [[buffer(2)]],
    constant int& C [[buffer(3)]],
    constant int& groups [[buffer(4)]],
    constant int& chunks [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint g [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpsz [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]) {
    int cg = C / groups;
    int c0 = (int)g * cg;
    int total = chunks * cg;
    float sum = 0.0f, sq = 0.0f;
    for (int i = tid; i < total; i += tpsz) {
        size_t idx = ((size_t)(i / cg) * C + c0 + (i % cg)) * 2;
        sum += partial[idx];
        sq += partial[idx + 1];
    }
    sum = simd_sum(sum);
    sq = simd_sum(sq);
    threadgroup float pSum[32], pSq[32];
    if (lane == 0) { pSum[sgid] = sum; pSq[sgid] = sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float ts = 0.0f, tq = 0.0f;
        uint ng = (tpsz + 31) / 32;
        for (uint i = 0; i < ng; i++) { ts += pSum[i]; tq += pSq[i]; }
        float n = (float)HW * (float)cg;
        float mean = ts / n;
        stats[g * 2] = mean;
        stats[g * 2 + 1] = rsqrt(tq / n - mean * mean + eps);
    }
}

// Pass 3: elementwise normalize + affine (+ optional SiLU), coalesced.
kernel void dv_gn_apply(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device const float* stats [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant int& C [[buffer(5)]],
    constant int& groups [[buffer(6)]],
    constant int& doSilu [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
    int c = gid.x;
    if (c >= C) return;
    int g = c / (C / groups);
    size_t idx = (size_t)gid.y * C + c;
    float v = (x[idx] - stats[g * 2]) * stats[g * 2 + 1] * gamma[c] + beta[c];
    if (doSilu != 0) { v = v / (1.0f + metal::exp(-v)); }
    y[idx] = v;
}

// Per-channel bias (+ optional SiLU), in place over [HW, C].
kernel void dv_bias_act(
    device float* x [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    constant int& C [[buffer(2)]],
    constant int& doSilu [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    int c = gid.x;
    if (c >= C) return;
    size_t idx = (size_t)gid.y * C + c;
    float v = x[idx] + bias[c];
    if (doSilu != 0) { v = v / (1.0f + metal::exp(-v)); }
    x[idx] = v;
}

// Nearest-neighbor 2× upsample, NHWC.
kernel void dv_upsample2(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant int& H [[buffer(2)]],
    constant int& W [[buffer(3)]],
    constant int& C [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
    int c = gid.x, w = gid.y, h = gid.z;
    if (c >= C || w >= 2 * W || h >= 2 * H) return;
    float v = x[((size_t)(h / 2) * W + (w / 2)) * C + c];
    y[((size_t)h * (2 * W) + w) * C + c] = v;
}

// y = a + b (f32).
kernel void dv_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    y[gid] = a[gid] + b[gid];
}

// scores[M,N] = Σk A[m,k]·B[n,k]  (Q·Kᵀ, both [rows, C])
kernel void dv_matmul_nt(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    int n = gid.x, m = gid.y;
    if (n >= N || m >= M) return;
    device const float* a = A + (size_t)m * K;
    device const float* b = B + (size_t)n * K;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc = fma(a[k], b[k], acc);
    Y[(size_t)m * N + n] = acc;
}

// Y[M,N] = Σk P[m,k]·V[k,n]  (probs · V)
kernel void dv_matmul_nn(
    device const float* P [[buffer(0)]],
    device const float* V [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    int n = gid.x, m = gid.y;
    if (n >= N || m >= M) return;
    device const float* p = P + (size_t)m * K;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc = fma(p[k], V[(size_t)k * N + n], acc);
    Y[(size_t)m * N + n] = acc;
}

// Row softmax with scale: y = softmax(x·scale), one threadgroup per row.
kernel void dv_softmax_rows(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpsz [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]) {
    device const float* xr = x + (size_t)row * N;
    device float* yr = y + (size_t)row * N;
    float mx = -INFINITY;
    for (int i = tid; i < N; i += tpsz) mx = max(mx, xr[i] * scale);
    mx = simd_max(mx);
    threadgroup float pMax[32], pSum[32];
    threadgroup float rowMax, rowSum;
    if (lane == 0) pMax[sgid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = -INFINITY;
        uint ng = (tpsz + 31) / 32;
        for (uint i = 0; i < ng; i++) t = max(t, pMax[i]);
        rowMax = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = rowMax;
    float s = 0.0f;
    for (int i = tid; i < N; i += tpsz) s += metal::exp(xr[i] * scale - m);
    s = simd_sum(s);
    if (lane == 0) pSum[sgid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        uint ng = (tpsz + 31) / 32;
        for (uint i = 0; i < ng; i++) t += pSum[i];
        rowSum = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invS = 1.0f / rowSum;
    for (int i = tid; i < N; i += tpsz) yr[i] = metal::exp(xr[i] * scale - m) * invS;
}
