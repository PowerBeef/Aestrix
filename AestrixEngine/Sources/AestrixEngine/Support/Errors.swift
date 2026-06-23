import Foundation

/// Errors thrown by ``AestrixEngine``.
public enum AestrixError: Error, Sendable, CustomStringConvertible {
    /// `load`/`generate`/`edit` called before the model is resident.
    case modelNotLoaded
    /// A milestone boundary not yet implemented.
    case notImplemented(String)
    /// A model file failed to download or failed size verification.
    case downloadFailed(String)
    /// Weights could not be loaded/quantized as expected.
    case weightLoadFailed(String)
    /// The OS asked us to release memory mid-run (memory pressure).
    case memoryPressure
    /// A parity check against the Python reference failed.
    case parityFailure(String)
    /// An image was the wrong size/format for the requested operation.
    case invalidInput(String)

    public var description: String {
        switch self {
        case .modelNotLoaded: "modelNotLoaded"
        case .notImplemented(let s): "notImplemented: \(s)"
        case .downloadFailed(let s): "downloadFailed: \(s)"
        case .weightLoadFailed(let s): "weightLoadFailed: \(s)"
        case .memoryPressure: "memoryPressure"
        case .parityFailure(let s): "parityFailure: \(s)"
        case .invalidInput(let s): "invalidInput: \(s)"
        }
    }
}
