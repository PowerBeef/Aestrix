import Foundation

/// Packaging check for the MLX metallib next to `imarello`.
///
/// Steel kernels are required for the product path. NAX kernels may be present
/// in a full metallib but are hardware-gated (M5 / A19) and never fail the product check.
public struct MetallibVerification: Sendable, Equatable, Codable {
    public var path: String
    public var byteCount: Int
    public var isStub: Bool
    public var steelSymbolsPresent: [String]
    public var steelSymbolsMissing: [String]
    public var naxSymbolsPresent: [String]
    public var naxSymbolsMissing: [String]
    public var productReady: Bool
    public var naxPackaged: Bool
    public var note: String

    public static let stubByteThreshold = 1_000_000

    public static let requiredSteelSymbols = [
        "steel_attention",
        "affine_qmm",
        "rms_norm",
    ]

    public static let optionalNAXSymbols = [
        "affine_qmm_n_nax",
        "steel_gemm_fused_nax",
        "quantized_nax",
    ]

    public static func verify(data: Data, path: String = "<memory>") -> MetallibVerification {
        let steelPresent = requiredSteelSymbols.filter { containsASCII($0, in: data) }
        let steelMissing = requiredSteelSymbols.filter { !containsASCII($0, in: data) }
        let naxPresent = optionalNAXSymbols.filter { containsASCII($0, in: data) }
        let naxMissing = optionalNAXSymbols.filter { !containsASCII($0, in: data) }
        let stub = data.count < stubByteThreshold
        let productReady = !stub && steelMissing.isEmpty
        let naxPackaged = !stub && naxMissing.isEmpty
        let note: String
        if stub {
            note = "stub metallib (\(data.count) bytes) — run Scripts/ensure-metallib.sh"
        } else if !productReady {
            note = "Steel symbols missing: \(steelMissing.joined(separator: ", "))"
        } else if naxPackaged {
            note = "full metallib: Steel ready; NAX kernels packaged (hardware-gated)"
        } else {
            note = "full metallib: Steel product path ready; NAX kernels optional/incomplete"
        }
        return MetallibVerification(
            path: path,
            byteCount: data.count,
            isStub: stub,
            steelSymbolsPresent: steelPresent,
            steelSymbolsMissing: steelMissing,
            naxSymbolsPresent: naxPresent,
            naxSymbolsMissing: naxMissing,
            productReady: productReady,
            naxPackaged: naxPackaged,
            note: note
        )
    }

    public static func verify(url: URL) -> MetallibVerification {
        guard let data = try? Data(contentsOf: url) else {
            return MetallibVerification(
                path: url.path,
                byteCount: 0,
                isStub: true,
                steelSymbolsPresent: [],
                steelSymbolsMissing: requiredSteelSymbols,
                naxSymbolsPresent: [],
                naxSymbolsMissing: optionalNAXSymbols,
                productReady: false,
                naxPackaged: false,
                note: "metallib missing at \(url.path)"
            )
        }
        return verify(data: data, path: url.path)
    }

    public static func resolveExisting(relativeTo executable: URL? = nil) -> URL? {
        var candidates: [URL] = []
        if let executable {
            let dir = executable.resolvingSymlinksInPath().deletingLastPathComponent()
            candidates += [
                dir.appendingPathComponent("mlx.metallib"),
                dir.appendingPathComponent("default.metallib"),
                dir.appendingPathComponent("Resources/mlx.metallib"),
            ]
        }
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        candidates += [
            cwd.appendingPathComponent("Tools/Metal/mlx.metallib"),
            cwd.appendingPathComponent(".build/release/mlx.metallib"),
            cwd.appendingPathComponent(".build/debug/mlx.metallib"),
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// App / framework bundles first (iOS host), then the CLI search path.
    public static func resolveFromBundles() -> URL? {
        let resourceNames = ["mlx", "default"]
        var bundles: [Bundle] = [Bundle.main]
        bundles.append(contentsOf: Bundle.allFrameworks)
        bundles.append(contentsOf: Bundle.allBundles)
        var seen = Set<String>()
        for bundle in bundles {
            let id = bundle.bundlePath
            if seen.contains(id) { continue }
            seen.insert(id)
            for name in resourceNames {
                if let url = bundle.url(forResource: name, withExtension: "metallib") {
                    return url
                }
            }
        }
        return resolveExisting()
    }

    private static func containsASCII(_ needle: String, in data: Data) -> Bool {
        guard let pattern = needle.data(using: .utf8), !pattern.isEmpty else { return false }
        return data.range(of: pattern) != nil
    }
}
