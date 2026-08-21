import Foundation
import Metal

/// Build-time Direct TE shader library. Half and bfloat variants are compiled
/// as distinct symbols into the same metallib.
enum DirectGlueKernels {
    static func makeLibrary(
        device: MTLDevice, directMetallibURL: URL
    ) throws -> MTLLibrary {
        try device.makeLibrary(URL: directMetallibURL)
    }

    static func functionName(_ base: String, dtypeName: String) -> String {
        switch dtypeName {
        case "half": return "\(base)_half"
        case "bfloat": return "\(base)_bfloat"
        default: return "\(base)_unsupported"
        }
    }
}
