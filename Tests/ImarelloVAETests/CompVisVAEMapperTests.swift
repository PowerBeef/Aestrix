import Testing
@testable import ImarelloVAE

@Suite("CompVis → Flux2Decoder Small Decoder map")
struct CompVisVAEMapperTests {
    @Test("up.N reverses onto diffusers up_blocks")
    func upIndexReverses() {
        #expect(CompVisVAEMapper.diffusersUpIndex(compVisUp: 0) == 3)
        #expect(CompVisVAEMapper.diffusersUpIndex(compVisUp: 3) == 0)
    }

    @Test("representative CompVis keys map onto Flux2Decoder names")
    func keyMap() {
        let samples: [(String, String)] = [
            ("conv_in.weight", "decoder.conv_in.weight"),
            ("conv_out.bias", "decoder.conv_out.bias"),
            ("norm_out.weight", "decoder.conv_norm_out.weight"),
            ("post_quant_conv.weight", "post_quant_conv.weight"),
            ("mid.block_1.conv1.weight", "decoder.mid_block.resnets.0.conv1.weight"),
            ("mid.block_2.norm2.bias", "decoder.mid_block.resnets.1.norm2.bias"),
            ("mid.attn_1.norm.weight", "decoder.mid_block.attentions.0.group_norm.weight"),
            ("mid.attn_1.q.weight", "decoder.mid_block.attentions.0.to_q.weight"),
            ("mid.attn_1.k.bias", "decoder.mid_block.attentions.0.to_k.bias"),
            ("mid.attn_1.v.weight", "decoder.mid_block.attentions.0.to_v.weight"),
            ("mid.attn_1.proj_out.weight", "decoder.mid_block.attentions.0.to_out.weight"),
            ("up.3.block.0.conv1.weight", "decoder.up_blocks.0.resnets.0.conv1.weight"),
            ("up.3.upsample.conv.weight", "decoder.up_blocks.0.upsamplers.0.conv.weight"),
            ("up.0.block.0.nin_shortcut.weight", "decoder.up_blocks.3.resnets.0.conv_shortcut.weight"),
            ("up.0.block.2.conv2.bias", "decoder.up_blocks.3.resnets.2.conv2.bias"),
            ("up.1.block.0.nin_shortcut.bias", "decoder.up_blocks.2.resnets.0.conv_shortcut.bias"),
        ]
        for (from, to) in samples {
            #expect(CompVisVAEMapper.mappedKey(from) == to, "\(from) → \(to)")
        }
        #expect(CompVisVAEMapper.mappedKey("encoder.conv_in.weight") == nil)
        #expect(CompVisVAEMapper.mappedKey("bn.running_mean") == nil)
    }

    @Test("all 140 BFL small_decoder key names map")
    func everyKnownKeyMaps() {
        for key in Self.bflSmallDecoderKeys {
            #expect(CompVisVAEMapper.mappedKey(key) != nil, "unmapped \(key)")
        }
    }

    /// Exact key list from BFL `small_decoder.safetensors` (2026-08-14 header dump).
    private static let bflSmallDecoderKeys: [String] = {
        var keys: [String] = [
            "conv_in.bias", "conv_in.weight",
            "conv_out.bias", "conv_out.weight",
            "norm_out.bias", "norm_out.weight",
            "post_quant_conv.bias", "post_quant_conv.weight",
        ]
        for leaf in ["bias", "weight"] {
            keys.append("mid.attn_1.k.\(leaf)")
            keys.append("mid.attn_1.q.\(leaf)")
            keys.append("mid.attn_1.v.\(leaf)")
            keys.append("mid.attn_1.proj_out.\(leaf)")
            keys.append("mid.attn_1.norm.\(leaf)")
            for block in ["block_1", "block_2"] {
                for name in ["conv1", "conv2", "norm1", "norm2"] {
                    keys.append("mid.\(block).\(name).\(leaf)")
                }
            }
        }
        // up.0 (output): 3 resnets, shortcut on resnet 0, no upsample
        for leaf in ["bias", "weight"] {
            keys.append("up.0.block.0.nin_shortcut.\(leaf)")
            for j in 0...2 {
                for name in ["conv1", "conv2", "norm1", "norm2"] {
                    keys.append("up.0.block.\(j).\(name).\(leaf)")
                }
            }
            // up.1: shortcut + upsample
            keys.append("up.1.block.0.nin_shortcut.\(leaf)")
            keys.append("up.1.upsample.conv.\(leaf)")
            for j in 0...2 {
                for name in ["conv1", "conv2", "norm1", "norm2"] {
                    keys.append("up.1.block.\(j).\(name).\(leaf)")
                }
            }
            // up.2 / up.3: no shortcut, have upsample
            for n in 2...3 {
                keys.append("up.\(n).upsample.conv.\(leaf)")
                for j in 0...2 {
                    for name in ["conv1", "conv2", "norm1", "norm2"] {
                        keys.append("up.\(n).block.\(j).\(name).\(leaf)")
                    }
                }
            }
        }
        return keys
    }()
}
