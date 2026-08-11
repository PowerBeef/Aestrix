#include <metal_stdlib>
using namespace metal;
// Placeholder default library for MLX device init.
// Real kernels are JIT-compiled from source by mlx-swift.
kernel void aestrix_mlx_noop(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
  if (id == 0) { out[0] = 0; }
}
