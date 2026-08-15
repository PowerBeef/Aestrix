import Foundation

/// Which VAE decoder graph to run at T2I / I2I decode time.
///
/// The encoder (I2I stage-0) is always the full FLUX.2 AE from the klein pack.
/// Do not load BFL `full_encoder_small_decoder.safetensors` as a second encoder.
/// Small Decoder is a **separate** distilled module (`[96, 192, 384, 384]` vs
/// `[128, 256, 512, 512]`) — not a weight swap into `Flux2Decoder`.
/// Product default stays `.full` until a quality A/B flips it.
public enum VAEDecoderVariant: String, Sendable, Codable, CaseIterable {
    /// Full klein-pack decoder (`block_out_channels` 128/256/512/512).
    case full = "full"
    /// BFL FLUX.2 Small Decoder (Apache-2.0, narrower channels).
    case smallDecoder = "small-decoder"

    /// Hub pin for `small_decoder.safetensors`. Keep in lockstep with `Docs/hub-pins.json`.
    public static let smallDecoderPin = HubPin(
        modelID: "black-forest-labs/FLUX.2-small-decoder",
        revision: "a3efc24f613ef42d9428af62fdbd6f5fd8856c4a"
    )

    /// Weight file inside the Small Decoder snapshot (not the 250 MB full-AE bundle).
    public static let smallDecoderFileName = "small_decoder.safetensors"

    public var blockOutChannels: [Int] {
        switch self {
        case .full: return [128, 256, 512, 512]
        case .smallDecoder: return [96, 192, 384, 384]
        }
    }

    public var downloadCommand: String {
        let dest =
            "~/Library/Caches/Imarello/models/"
            + Self.smallDecoderPin.modelID.replacingOccurrences(of: "/", with: "--")
        return
            "hf download \(Self.smallDecoderPin.modelID) --revision \(Self.smallDecoderPin.revision) "
            + "--include \(Self.smallDecoderFileName) --include config.json --local-dir \(dest)"
    }
}
