import Foundation

public enum AestrixError: Error, Sendable, LocalizedError {
    case invalidDimensions(width: Int, height: Int, reason: String)
    case resolutionExceedsTier(side: Int, tier: DeviceTier, maxSide: Int)
    case moduleAlreadyLoaded(String)
    case moduleNotLoaded(String)
    case concurrentGenerationNotAllowed
    case cancelled
    case outOfMemory(stage: String, detail: String)
    case weightsNotFound(modelID: String, path: String?)
    case unsupportedWeightFormat(String)
    case notImplemented(String)
    case imageLoadFailed(path: String, reason: String)
    case invalidStrength(Float)

    public var errorDescription: String? {
        switch self {
        case .invalidDimensions(let w, let h, let reason):
            return "Invalid dimensions \(w)×\(h): \(reason)"
        case .resolutionExceedsTier(let side, let tier, let maxSide):
            return "Side \(side) exceeds tier \(tier.rawValue) max \(maxSide)"
        case .moduleAlreadyLoaded(let name):
            return "Module already loaded: \(name)"
        case .moduleNotLoaded(let name):
            return "Module not loaded: \(name)"
        case .concurrentGenerationNotAllowed:
            return "Only one generation may run at a time on this pipeline"
        case .cancelled:
            return "Generation cancelled"
        case .outOfMemory(let stage, let detail):
            return "Out of memory during \(stage): \(detail)"
        case .weightsNotFound(let id, let path):
            return "Weights not found for \(id)" + (path.map { " at \($0)" } ?? "")
        case .unsupportedWeightFormat(let detail):
            return "Unsupported weight format: \(detail)"
        case .notImplemented(let feature):
            return "Not implemented: \(feature)"
        case .imageLoadFailed(let path, let reason):
            return "Failed to load image at \(path): \(reason)"
        case .invalidStrength(let s):
            return "Strength must be in (0, 1], got \(s)"
        }
    }
}
