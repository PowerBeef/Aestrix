import Foundation

/// Make MLX kernels loadable inside the xctest bundle.
///
/// The nojit mlx-swift fork resolves every kernel by name from the full
/// metallib, and MLX's colocated lookup searches next to the *test* binary
/// (`ImarelloPackageTests.xctest/Contents/MacOS/`), not the products dir where
/// `Scripts/ensure-metallib.sh` installs it. Clone it in before the first eval;
/// on APFS the copy is instant. If only the `swift build` stub exists, the copy
/// still happens and MLX fails the same way the CLI would — run the script.
enum TestMetallib {
    @discardableResult
    static func install() -> Bool { installed }

    private static let installed: Bool = {
        let fm = FileManager.default
        guard
            let exe = Bundle(for: BundleToken.self).executableURL?
                .resolvingSymlinksInPath()
        else { return false }
        let dest = exe.deletingLastPathComponent()
            .appendingPathComponent("mlx.metallib")
        // Products dir: …/debug/ImarelloPackageTests.xctest/Contents/MacOS/exe
        let products = exe.deletingLastPathComponent()  // MacOS
            .deletingLastPathComponent()  // Contents
            .deletingLastPathComponent()  // .xctest
            .deletingLastPathComponent()  // debug
        let src = products.appendingPathComponent("mlx.metallib")
        guard let srcSize = try? fm.attributesOfItem(atPath: src.path)[.size] as? Int64
        else { return fm.fileExists(atPath: dest.path) }
        if let destSize = try? fm.attributesOfItem(atPath: dest.path)[.size] as? Int64,
            destSize == srcSize
        {
            return true
        }
        try? fm.removeItem(at: dest)
        do {
            try fm.copyItem(at: src, to: dest)
            return true
        } catch {
            return false
        }
    }()

    private final class BundleToken {}
}
