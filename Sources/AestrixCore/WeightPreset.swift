/// Product weights are always pre-quantized. There is no bf16 product preset.
public enum WeightPreset: String, Sendable, Codable, CaseIterable {
    case bits3 = "3bit"
    case bits4 = "4bit"
    case bits6 = "6bit"
    case bits8 = "8bit"

    /// Default Hugging Face repo for this preset (see Docs/WEIGHTS.md).
    public var defaultModelID: String {
        switch self {
        case .bits3: return "mlx-community/FLUX.2-Klein-4B-3bit"
        case .bits4: return "mlx-community/FLUX.2-Klein-4B-4bit"
        case .bits6: return "mlx-community/FLUX.2-Klein-4B-6bit"
        case .bits8: return "mlx-community/flux2-klein-4b-8bit"
        }
    }
}
