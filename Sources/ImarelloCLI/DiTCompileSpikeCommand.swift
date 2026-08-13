import ArgumentParser
import Foundation
import ImarelloCore
import ImarelloDiT
import ImarelloRuntime

/// PERF.md S1 research spike: block-level `MLX.compile` under a resident DiT.
struct DiTCompileSpike: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "dit-compile-spike",
        abstract: "Research: time one DiT block uncompiled vs MLX.compile (resident-DiT scenario). See Docs/PERF.md S1."
    )

    @Option(help: "Square canvas side in pixels (packed side = side/16).")
    var width: Int = 512

    @Option(help: "Text tokens in the joint sequence.")
    var textSeq: Int = 512

    @Option(help: "Timed iterations per variant (after 1 warmup).")
    var iters: Int = 6

    @Flag(name: .long, help: "Run on 8 GB-class hosts (not recommended; GPU compile can starve WindowServer).")
    var force: Bool = false

    func run() async throws {
        try ensureMLXReady()
        let config = ImarelloConfig.autoDetectingTier()
        if config.tier == .low, !force {
            print("error: dit-compile-spike loads a full DiT and compiles Metal on-device.")
            print("  Refused on 8 GB-class hosts (WindowServer watchdog risk). Pass --force if isolated.")
            throw ExitCode.failure
        }
        let pipeline = ImarelloPipeline(config: config)
        guard let snapshotPath = await pipeline.snapshotPath else {
            print("error: no local snapshot for \(config.modelID)")
            throw ExitCode.failure
        }
        let transformerDir = URL(fileURLWithPath: snapshotPath)
            .appendingPathComponent("transformer", isDirectory: true)

        print("loading DiT (4-bit) from \(transformerDir.path) …")
        let model = try TransformerWeights.loadQuantized(from: transformerDir)

        let imageSide = width / 16
        print("spike: canvas=\(width)² image_seq=\(imageSide * imageSide) text_seq=\(textSeq) iters=\(iters)")
        let report = BlockCompileSpike.run(
            model: model, textSeq: textSeq, imageSide: imageSide, iters: iters)

        func show(_ timings: [BlockCompileSpike.Timing], firstCallMs: Double, count: Int) {
            guard let product = timings.first(where: { $0.label.hasSuffix("product") }) else { return }
            for t in timings {
                let delta = (t.msPerIter - product.msPerIter) / product.msPerIter * 100
                print(String(
                    format: "  %-18@ %8.1f ms/iter  (%+.1f%% vs product; ×%d blocks ≈ %.0f ms/step)",
                    t.label as NSString, t.msPerIter, delta, count, t.msPerIter * Double(count)))
            }
            print(String(format: "  compile first call: %.0f ms", firstCallMs))
        }
        print("single-stream block (×20 per step):")
        show(report.singleBlock, firstCallMs: report.compileFirstCallMsSingle, count: 20)
        print("double-stream block (×5 per step):")
        show(report.doubleBlock, firstCallMs: report.compileFirstCallMsDouble, count: 5)

        if let compiled = report.singleBlock.first(where: { $0.label.hasSuffix("compiled") }),
           let product = report.singleBlock.first(where: { $0.label.hasSuffix("product") })
        {
            let win = (product.msPerIter - compiled.msPerIter) / product.msPerIter * 100
            let verdict = win >= 10
                ? "GO — ≥10% block win; consider full-forward compile under resident DiT"
                : "NO-GO — <10% block win; keep full compile parked (PERF.md S1)"
            print(String(format: "verdict: %.1f%% single-block win → %@", win, verdict))
        }
    }
}
