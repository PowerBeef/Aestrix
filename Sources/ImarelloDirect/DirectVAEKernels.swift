import Foundation
import Metal

/// Build-time Direct VAE shader library.
enum DirectVAEKernels {
    static func makeLibrary(
        device: MTLDevice, directMetallibURL: URL
    ) throws -> MTLLibrary {
        try device.makeLibrary(URL: directMetallibURL)
    }
}
