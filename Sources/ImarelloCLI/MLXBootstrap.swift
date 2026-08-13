import Foundation

/// Ensure MLX can find a default Metal library when built via `swift build`.
///
/// `swift build` does not emit mlx-swift's default metallib resource bundle.
/// MLX looks next to the executable for `mlx.metallib` (and a few other paths).
/// We ship a tiny placeholder metallib as a package resource and copy it next to
/// the running binary on first launch (before any MLX op).
enum MLXBootstrap {
    static func ensureMetallibBesideExecutable() {
        let exe = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
        let binDir = exe.deletingLastPathComponent()
        let candidates = [
            binDir.appendingPathComponent("mlx.metallib"),
            binDir.appendingPathComponent("default.metallib"),
            binDir.appendingPathComponent("Resources/mlx.metallib"),
        ]
        if candidates.contains(where: { FileManager.default.fileExists(atPath: $0.path) }) {
            return
        }

        guard let resource = Bundle.module.url(forResource: "mlx", withExtension: "metallib") else {
            fputs(
                "warning: mlx.metallib resource missing; MLX Metal init may fail. "
                    + "Run Scripts/ensure-metallib.sh after swift build.\n",
                stderr
            )
            return
        }

        let dest = candidates[0]
        do {
            try FileManager.default.copyItem(at: resource, to: dest)
            // Also seed Resources/ and default.metallib for alternate search paths.
            let resourcesDir = binDir.appendingPathComponent("Resources", isDirectory: true)
            try? FileManager.default.createDirectory(at: resourcesDir, withIntermediateDirectories: true)
            let resDest = resourcesDir.appendingPathComponent("mlx.metallib")
            if !FileManager.default.fileExists(atPath: resDest.path) {
                try? FileManager.default.copyItem(at: resource, to: resDest)
            }
            let defaultDest = binDir.appendingPathComponent("default.metallib")
            if !FileManager.default.fileExists(atPath: defaultDest.path) {
                try? FileManager.default.copyItem(at: resource, to: defaultDest)
            }
        } catch {
            fputs("warning: failed to install mlx.metallib: \(error)\n", stderr)
        }
    }
}
