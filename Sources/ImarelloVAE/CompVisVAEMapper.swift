import Foundation
import MLX
import ImarelloCore

/// Maps BFL `small_decoder.safetensors` (CompVis / LDM names, PyTorch conv layout)
/// onto `Flux2Decoder` + `post_quant_conv` keys (diffusers names, MLX OHWI convs).
///
/// BFL file has no `bn.*` — packed-latent BN stats stay on the klein pack.
public enum CompVisVAEMapper {
    /// Diffusers up-block index for a CompVis `up.{n}` block (n is output-side).
    public static func diffusersUpIndex(compVisUp: Int) -> Int {
        3 - compVisUp
    }

    /// Rename one CompVis key. Returns nil if the key is unknown.
    public static func mappedKey(_ compVis: String) -> String? {
        if compVis == "post_quant_conv.weight" || compVis == "post_quant_conv.bias" {
            return compVis
        }
        if compVis.hasPrefix("conv_in.") {
            return "decoder." + compVis
        }
        if compVis.hasPrefix("conv_out.") {
            return "decoder." + compVis
        }
        if let rest = suffix(compVis, prefix: "norm_out.") {
            return "decoder.conv_norm_out." + rest
        }
        if let rest = suffix(compVis, prefix: "mid.block_1.") {
            return "decoder.mid_block.resnets.0." + rest
        }
        if let rest = suffix(compVis, prefix: "mid.block_2.") {
            return "decoder.mid_block.resnets.1." + rest
        }
        if let rest = suffix(compVis, prefix: "mid.attn_1.") {
            return mapMidAttention(rest)
        }
        if let parsed = parseUpKey(compVis) {
            return parsed
        }
        return nil
    }

    /// Convert one tensor: PyTorch `[O,I,H,W]` convs → MLX `[O,H,W,I]`;
    /// mid-attn 1×1 convs squeeze to Linear `[O,I]`.
    public static func convertWeight(compVisKey: String, array: MLXArray) -> MLXArray {
        guard array.ndim == 4 else { return array }
        if isMidAttentionProjection(compVisKey),
           array.dim(2) == 1, array.dim(3) == 1
        {
            return array.reshaped([array.dim(0), array.dim(1)])
        }
        // [O, I, H, W] → [O, H, W, I]
        return array.transposed(0, 2, 3, 1)
    }

    public static func remap(_ arrays: [String: MLXArray]) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(arrays.count)
        var unknown: [String] = []
        for (key, value) in arrays {
            guard let mapped = mappedKey(key) else {
                unknown.append(key)
                continue
            }
            out[mapped] = convertWeight(compVisKey: key, array: value)
        }
        if !unknown.isEmpty {
            throw ImarelloError.unsupportedWeightFormat(
                "unmapped Small Decoder keys: \(unknown.sorted().joined(separator: ", "))")
        }
        return out
    }

    // MARK: - private

    private static func suffix(_ key: String, prefix: String) -> String? {
        guard key.hasPrefix(prefix) else { return nil }
        return String(key.dropFirst(prefix.count))
    }

    private static func mapMidAttention(_ rest: String) -> String? {
        let pairs: [(String, String)] = [
            ("norm.", "group_norm."),
            ("q.", "to_q."),
            ("k.", "to_k."),
            ("v.", "to_v."),
            ("proj_out.", "to_out."),
        ]
        for (from, to) in pairs {
            if let leaf = suffix(rest, prefix: from) {
                return "decoder.mid_block.attentions.0." + to + leaf
            }
        }
        return nil
    }

    private static func isMidAttentionProjection(_ key: String) -> Bool {
        key.hasPrefix("mid.attn_1.q.")
            || key.hasPrefix("mid.attn_1.k.")
            || key.hasPrefix("mid.attn_1.v.")
            || key.hasPrefix("mid.attn_1.proj_out.")
    }

    /// `up.{n}.block.{j}.nin_shortcut.X` / `up.{n}.block.{j}.Y` / `up.{n}.upsample.conv.X`
    private static func parseUpKey(_ key: String) -> String? {
        guard key.hasPrefix("up.") else { return nil }
        let parts = key.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
        // up N block J ...  OR  up N upsample conv ...
        guard parts.count >= 4, let n = Int(parts[1]) else { return nil }
        let up = diffusersUpIndex(compVisUp: n)
        if parts[2] == "upsample" {
            // up.N.upsample.conv.weight
            let rest = parts.dropFirst(3).joined(separator: ".")
            return "decoder.up_blocks.\(up).upsamplers.0." + rest
        }
        if parts[2] == "block", parts.count >= 5, let j = Int(parts[3]) {
            var leafParts = Array(parts.dropFirst(4))
            if leafParts.first == "nin_shortcut" {
                leafParts[0] = "conv_shortcut"
            }
            return "decoder.up_blocks.\(up).resnets.\(j)." + leafParts.joined(separator: ".")
        }
        return nil
    }
}
