import Foundation
import ImarelloCore

/// Fail closed unless a complete no-JIT metallib is installed beside the
/// executable by `Scripts/ensure-metallib.sh`.
enum MLXBootstrap {
    static func verifyMetallibBesideExecutable() throws {
        let exe = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
        guard let url = MetallibVerification.resolveExisting(relativeTo: exe) else {
            throw ImarelloError.metallibNotReady(
                "full no-JIT mlx.metallib is missing; run Scripts/ensure-metallib.sh")
        }
        let check = MetallibVerification.verify(url: url)
        guard check.productReady else {
            throw ImarelloError.metallibNotReady(check.note)
        }
    }
}
