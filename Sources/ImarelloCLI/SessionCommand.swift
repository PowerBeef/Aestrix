import ArgumentParser
import Foundation
import ImarelloCore
import ImarelloWeights
import ImarelloRuntime

/// Warm interactive session: read prompts from stdin, keep modules resident across
/// generations (Tier-gated), so repeat generations pay only denoise + VAE decode.
struct Session: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "session",
        abstract: "Warm multi-prompt session: modules stay resident between generations (high-RAM; use --force-resident to override)."
    )

    @Option var width: Int = 512
    @Option var height: Int = 512
    @Option var steps: Int = 4
    @Option var seed: UInt64?

    @Option(name: .long, help: "4bit (the product lock; 6/8-bit pins are non-product references)")
    var weights: String = "4bit"

    @Option(name: .long, help: "Output directory (default: ~/Library/Caches/Imarello/outputs)")
    var outputDir: String?

    @Option(name: .long, help: "Text tokens to DiT: 512 (default, padded product path) | auto (trim pad; faster, weaker conditioning). Docs/TEXT_TOKENS.md.")
    var textTokens: String = "512"

    @Option(name: .long, help: "VAE decode graph: small-decoder (default) | full (klein pack).")
    var vaeVariant: String = "small-decoder"

    @Flag(name: .long, inversion: .prefixedNo, help: "Cache prompt embeddings on disk (default on; keeps TE out of the resident set on repeats).")
    var embedCache: Bool = true

    @Flag(name: .long, help: "Keep modules resident even below the 16 GB RAM gate (risky on 8 GB at 1024²).")
    var forceResident: Bool = false

    func run() async throws {
        try ensureMLXReady()
        try validateSteps(steps)
        guard let preset = WeightPreset(rawValue: weights) else {
            throw ValidationError("Unknown weights preset: \(weights)")
        }
        guard preset == .bits4 else {
            throw ValidationError(
                "--weights \(weights) is not a product path (4-bit lock, Docs/WEIGHTS.md); "
                    + "the 6/8-bit pins are reference-only and cannot load end-to-end")
        }
        guard let textTokenMode = TextTokenMode(rawValue: textTokens) else {
            throw ValidationError("Unknown --text-tokens '\(textTokens)'; use 512 | auto")
        }

        let physGB = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824.0
        let resident = forceResident || physGB >= 15.0
        var config = ImarelloConfig.autoDetectingTier()
        config.apply(preset: preset)
        config.memoryPolicy = resident ? .resident : .staged
        try applyVAEVariant(vaeVariant, to: &config)
        if config.vaeDecoderVariant == .smallDecoder,
           ModelPaths.resolveSmallDecoderIfPresent(config: config) == nil
        {
            print("error: Small Decoder snapshot missing")
            print("hint: \(VAEDecoderVariant.smallDecoder.downloadCommand)")
            throw ExitCode.failure
        }

        let pipeline = ImarelloPipeline(config: config)
        guard await pipeline.hasLocalSnapshot else {
            print("error: no local snapshot for \(config.modelID)")
            throw ExitCode.failure
        }

        print("session policy=\(config.memoryPolicy.rawValue) canvas=\(width)x\(height) steps=\(steps) embed-cache=\(embedCache ? "on" : "off") text-tokens=\(textTokenMode.rawValue)")
        if !resident {
            print("note: <16 GB RAM — staged policy (weights reload each prompt). Use --force-resident to keep modules warm; with --embed-cache the resident set is DiT+VAE (~2.1 GB).")
        }
        print("Enter a prompt per line (empty line or Ctrl-D to exit).")

        let outDirURL = outputDir.map { URL(fileURLWithPath: $0, isDirectory: true) }
        if let outDirURL {
            try FileManager.default.createDirectory(at: outDirURL, withIntermediateDirectories: true)
        }

        var index = 0
        while true {
            print("prompt> ", terminator: "")
            guard let line = readLine(strippingNewline: true) else { break }
            let prompt = line.trimmingCharacters(in: .whitespaces)
            if prompt.isEmpty || prompt == "exit" || prompt == "quit" { break }

            index += 1
            let outURL = outDirURL.map {
                $0.appendingPathComponent("session_\(index)_s\(seed.map(String.init) ?? "r").png")
            }
            let request = T2IRequest(
                prompt: prompt,
                width: width,
                height: height,
                steps: steps,
                guidance: 1.0,
                seed: seed,
                outputURL: outURL,
                textTokens: textTokenMode,
                embedCache: embedCache
            )
            let start = Date()
            do {
                let url = try await pipeline.generate(request) { progress in
                    if progress.phase == .denoising {
                        print("  denoising step=\(progress.step + 1)/\(progress.totalSteps)")
                    }
                }
                let elapsed = Date().timeIntervalSince(start)
                print(String(format: "  wrote %@ (%.1f s)", url.path, elapsed))
            } catch {
                print("  error: \(error)")
            }
        }
        await pipeline.purge()
        print("session ended (\(index) generation\(index == 1 ? "" : "s"))")
    }
}
