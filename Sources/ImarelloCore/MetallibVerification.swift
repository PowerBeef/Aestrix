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
    /// The complete mlx-swift 0.32.1 no-JIT pack is roughly 155 MB. Keep a
    /// conservative floor well above the thin 3–5 MB JIT-era bundle so a
    /// partial library cannot be mistaken for the runtime artifact.
    public static let fullPackByteThreshold = 100_000_000

    /// mlx-swift revision stamped by `Scripts/ensure-metallib.sh`, when the
    /// artifact was installed through the reproducible repository recipe.
    public static func recordedRevision(for url: URL) -> String? {
        let revisionURL = url.appendingPathExtension("rev")
        guard let value = try? String(contentsOf: revisionURL, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !value.isEmpty
        else { return nil }
        return value
    }

    /// Names that must appear in the no-JIT MLX metallib. The product uses
    /// prequantized 4-bit weights, so packaged affine QMM kernels are mandatory.
    public static let requiredSteelSymbols = [
        "steel_attention",
        "rms_norm",
        "affine_qmm",
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
        let productReady = data.count >= fullPackByteThreshold && steelMissing.isEmpty
        let naxPackaged = !stub && naxMissing.isEmpty
        let note: String
        if stub {
            note = "stub metallib (\(data.count) bytes) — run Scripts/ensure-metallib.sh"
        } else if !steelMissing.isEmpty {
            note = "Steel symbols missing: \(steelMissing.joined(separator: ", "))"
        } else if data.count < fullPackByteThreshold {
            note = "partial/JIT-era metallib (\(data.count) bytes) — full no-JIT pack required"
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
        return pickBest(among: candidates)
    }

    /// SPM resource-bundle names mlx-swift embeds as `SWIFTPM_BUNDLE`.
    public static let cmlxResourceBundleNames = [
        "mlx-swift_Cmlx",
        "mlx_Cmlx",
    ]

    /// App / framework bundles first (iOS host), then the CLI search path.
    ///
    /// `mlx-swift_Cmlx.bundle` is a **resource** bundle (`BNDL`) sitting next to
    /// the executable. It is *not* in `Bundle.allBundles` until something loads
    /// it. Asking every loaded bundle for `default.metallib` therefore either
    /// hits a UIKit/SwiftUI stub (~157 KB) or finds nothing. Walk the app
    /// wrapper on disk the same way Cmlx `try_load_bundle` does.
    public static func resolveFromBundles() -> URL? {
        var candidates: [URL] = []
        candidates += onDiskMetallibs(in: Bundle.main.bundleURL)
        if let exe = Bundle.main.executableURL {
            let dir = exe.resolvingSymlinksInPath().deletingLastPathComponent()
            candidates += onDiskMetallibs(in: dir)
            if let extra = resolveExisting(relativeTo: exe) {
                candidates.append(extra)
            }
        }
        if let extra = resolveExisting() {
            candidates.append(extra)
        }
        if let ready = pickBest(among: candidates) {
            return ready
        }

        // Last resort: already-loaded bundles (Mac CLI / test host).
        let resourceNames = ["mlx", "default"]
        var bundles: [Bundle] = [Bundle.main]
        bundles.append(contentsOf: Bundle.allFrameworks)
        bundles.append(contentsOf: Bundle.allBundles)
        var seen = Set<String>()
        var loaded: [URL] = []
        for bundle in bundles {
            let id = bundle.bundlePath
            if seen.contains(id) { continue }
            seen.insert(id)
            for name in resourceNames {
                if let url = bundle.url(forResource: name, withExtension: "metallib") {
                    loaded.append(url)
                }
            }
        }
        return pickBest(among: loaded)
    }

    /// Metallibs sitting in an `.app` / executable directory, including the
    /// mlx-swift Cmlx SPM resource bundle. Does not recurse into Frameworks
    /// system content.
    public static func onDiskMetallibs(in root: URL) -> [URL] {
        let fm = FileManager.default
        var urls: [URL] = []
        let fileNames = ["mlx.metallib", "default.metallib"]

        func appendIfPresent(_ url: URL) {
            if fm.fileExists(atPath: url.path) {
                urls.append(url)
            }
        }

        for name in fileNames {
            appendIfPresent(root.appendingPathComponent(name))
            appendIfPresent(root.appendingPathComponent("Resources").appendingPathComponent(name))
        }

        for bundleName in cmlxResourceBundleNames {
            if let bundleURL = Bundle(url: root)?.url(forResource: bundleName, withExtension: "bundle")
                ?? optionalDirectory(root.appendingPathComponent("\(bundleName).bundle"))
            {
                for name in fileNames {
                    appendIfPresent(bundleURL.appendingPathComponent(name))
                    appendIfPresent(
                        bundleURL.appendingPathComponent("Contents/Resources").appendingPathComponent(name)
                    )
                }
            }
        }

        if let children = try? fm.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) {
            for child in children where child.pathExtension == "bundle" {
                for name in fileNames {
                    appendIfPresent(child.appendingPathComponent(name))
                    appendIfPresent(
                        child.appendingPathComponent("Contents/Resources").appendingPathComponent(name)
                    )
                }
            }
        }
        return urls
    }

    private static func optionalDirectory(_ url: URL) -> URL? {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
            return nil
        }
        return url
    }

    /// Choose a product-ready no-JIT MLX metallib from a list. Partial or
    /// JIT-era artifacts are deliberately never returned.
    public static func pickBest(among urls: [URL]) -> URL? {
        var unique: [URL] = []
        var seen = Set<String>()
        for url in urls {
            let path = url.resolvingSymlinksInPath().path
            if seen.contains(path) { continue }
            seen.insert(path)
            unique.append(url)
        }
        guard !unique.isEmpty else { return nil }
        let verified = unique.map { url -> (URL, MetallibVerification) in
            (url, verify(url: url))
        }
        let ready = verified.filter { $0.1.productReady }
        if let best = ready.max(by: { $0.1.byteCount < $1.1.byteCount }) {
            return best.0
        }
        return nil
    }

    private static func containsASCII(_ needle: String, in data: Data) -> Bool {
        guard let pattern = needle.data(using: .utf8), !pattern.isEmpty else { return false }
        return data.range(of: pattern) != nil
    }
}
