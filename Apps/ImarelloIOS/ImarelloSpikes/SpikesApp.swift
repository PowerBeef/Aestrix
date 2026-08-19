import SwiftUI
import ImarelloDirect

/// Headless-ish spike runner: proves direct-engine kernel ABIs on the physical
/// device (the N-track's A19 NAX gate). Runs on launch, shows the report, and
/// writes it to Documents/nax-spike-report.txt for the host to pull:
///
///   xcrun devicectl device copy from --device <id> \
///     --domain-type appDataContainer --domain-identifier app.imarello.spikes \
///     --source Documents/nax-spike-report.txt --destination /tmp/nax-report.txt
@main
struct SpikesApp: App {
    @State private var report = "running qmm-nax spike…"

    var body: some Scene {
        WindowGroup {
            ScrollView {
                Text(report)
                    .font(.system(.caption2, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
            .task {
                // Keep the screen awake while foregrounded (low-battery lock
                // timers would otherwise suspend the run mid-spike).
                UIApplication.shared.isIdleTimerDisabled = true
                let result = await Task.detached(priority: .userInitiated) {
                    SpikeRunner.runAll()
                }.value
                report = result
                SpikeRunner.persist(result)
                UIApplication.shared.isIdleTimerDisabled = false
            }
        }
    }
}

enum SpikeRunner {
    /// The Cmlx resource bundle Xcode embeds alongside the app carries the
    /// full MLX metallib (NAX kernels included since the 26.2 floor).
    static func metallibURL() -> URL? {
        guard let bundleURL = Bundle.main.url(forResource: "mlx-swift_Cmlx", withExtension: "bundle"),
            let cmlx = Bundle(url: bundleURL)
        else { return nil }
        return cmlx.url(forResource: "default", withExtension: "metallib")
    }

    static func runAll() -> String {
        var out = "imarello device spikes — \(Date())\n\n"
        guard let metallib = metallibURL() else {
            return out + "FATAL: mlx-swift_Cmlx.bundle/default.metallib not found in app bundle"
        }
        out += "metallib: \(metallib.lastPathComponent) ✓\n\n"
        do {
            out += try DirectNAXQmmSpike.run(ditDirectory: nil, metallibURL: metallib)
        } catch {
            out += "qmm-nax spike FAILED: \(error)"
        }
        return out
    }

    static func persist(_ report: String) {
        guard let docs = FileManager.default.urls(
            for: .documentDirectory, in: .userDomainMask).first else { return }
        try? report.write(
            to: docs.appendingPathComponent("nax-spike-report.txt"),
            atomically: true, encoding: .utf8)
    }
}
