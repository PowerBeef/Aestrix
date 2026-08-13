import Foundation

/// Product weights are always pre-quantized. There is no bf16 product preset.
public enum WeightPreset: String, Sendable, Codable, CaseIterable {
    case bits3 = "3bit"
    case bits4 = "4bit"
    case bits6 = "6bit"
    case bits8 = "8bit"

    /// Default Hugging Face repo for this preset (see Docs/WEIGHTS.md, Docs/hub-pins.json).
    public var defaultModelID: String { pin.modelID }

    /// Hugging Face git commit this preset is validated against.
    public var pinnedRevision: String { pin.revision }

    /// Repo id + commit SHA. Keep in lockstep with `Docs/hub-pins.json`.
    public var pin: HubPin {
        switch self {
        case .bits3:
            return HubPin(
                modelID: "mlx-community/FLUX.2-Klein-4B-3bit",
                revision: "246946064c7218227b1e99509245392cdcedc9d3"
            )
        case .bits4:
            return HubPin(
                modelID: "mlx-community/FLUX.2-Klein-4B-4bit",
                revision: "1cebb9b45c21ece14a42615b16bf5fa4de9b56da"
            )
        case .bits6:
            return HubPin(
                modelID: "mlx-community/FLUX.2-Klein-4B-6bit",
                revision: "76fd8a876cb61126fb1fdce97eb9464eab063ff5"
            )
        case .bits8:
            return HubPin(
                modelID: "mlx-community/flux2-klein-4b-8bit",
                revision: "9beac1a3ad296d9e5e3f8845674e6577fa8654ec"
            )
        }
    }
}

/// Pinned Hugging Face snapshot (repo + commit).
public struct HubPin: Sendable, Hashable, Codable {
    public var modelID: String
    public var revision: String

    public init(modelID: String, revision: String) {
        self.modelID = modelID
        self.revision = revision
    }

    public var isCommitSHA: Bool {
        revision.count == 40 && revision.unicodeScalars.allSatisfy { Self.hex.contains($0) }
    }

    private static let hex = CharacterSet(charactersIn: "0123456789abcdefABCDEF")
}
