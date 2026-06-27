import Foundation
import MLX

/// Builds the 4D RoPE coordinate tensors (t, h, w, l) for text and image tokens.
///
/// Text tokens: (t=0, h=0, w=0, l=0..seq-1) — positioned along the layer axis.
/// Image tokens: (t=0, h=0..h-1, w=0..w-1, l=0) — positioned on a 2D spatial grid.
/// (mflux `Flux2PromptEncoder.prepare_text_ids` + `Flux2LatentCreator.prepare_grid_ids`.)
enum CoordinateBuilder {

    /// Text IDs: [seqLen, 4] — each row = (0, 0, 0, token_index).
    static func textIds(seqLen: Int = 512) -> MLXArray {
        var arr: [Int32] = []
        arr.reserveCapacity(seqLen * 4)
        for i in 0..<seqLen {
            arr += [0, 0, 0, Int32(i)]
        }
        return MLXArray(arr, [seqLen, 4])
    }

    /// Image IDs: [h*w, 4] — cartesian product, each row = (0, h_idx, w_idx, 0).
    static func imageIds(height: Int, width: Int, tCoord: Int = 0) -> MLXArray {
        var arr: [Int32] = []
        arr.reserveCapacity(height * width * 4)
        for h in 0..<height {
            for w in 0..<width {
                arr += [Int32(tCoord), Int32(h), Int32(w), 0]
            }
        }
        return MLXArray(arr, [height * width, 4])
    }
}
