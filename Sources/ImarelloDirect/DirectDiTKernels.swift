import Foundation
import Metal

/// Build-time Direct DiT shader library. Production code never compiles MSL
/// source at runtime; `Scripts/ensure-direct-metallib.sh` owns compilation.
enum DirectDiTKernels {
    static func makeLibrary(
        device: MTLDevice, directMetallibURL: URL
    ) throws -> MTLLibrary {
        try device.makeLibrary(URL: directMetallibURL)
    }
}
