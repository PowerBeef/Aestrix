import Foundation
import MLX
import ImarelloCore
import ImarelloText
import ImarelloDiT
import ImarelloVAE

/// Ensures at most one heavy module is loaded under staged policies.
/// Owned exclusively by `ImarelloPipeline` (actor); not shared across tasks.
public final class StageOrchestrator: @unchecked Sendable {
    public let textEncoder: TextEncoderModule
    public let dit: DiTModule
    public let vae: VAEModule
    public let memoryPolicy: MemoryPolicy
    public let probe: MemoryProbe

    public init(
        textEncoder: TextEncoderModule,
        dit: DiTModule,
        vae: VAEModule,
        memoryPolicy: MemoryPolicy,
        probe: MemoryProbe = MemoryProbe()
    ) {
        self.textEncoder = textEncoder
        self.dit = dit
        self.vae = vae
        self.memoryPolicy = memoryPolicy
        self.probe = probe
        // ImarelloCore does not link MLX; give its probes real MLX numbers here.
        if MemoryProbe.mlxSampler == nil {
            MemoryProbe.mlxSampler = { (UInt64(Memory.activeMemory), UInt64(Memory.cacheMemory)) }
        }
    }

    public func loadedModuleNames() -> [String] {
        var names: [String] = []
        if textEncoder.isLoaded { names.append(textEncoder.moduleName) }
        if dit.isLoaded { names.append(dit.moduleName) }
        if vae.isLoaded { names.append(vae.moduleName) }
        return names
    }

    public func assertStagedInvariant() throws {
        let loaded = loadedModuleNames()
        if memoryPolicy == .resident { return }
        if loaded.count > 1 {
            throw ImarelloError.outOfMemory(
                stage: "invariant",
                detail: "Staged policy violated: multiple modules loaded \(loaded)"
            )
        }
    }

    // MARK: - Exclusive load / unload (call from pipeline actor only)

    public func loadTextEncoderExclusive() async throws {
        if memoryPolicy != .resident {
            await unloadOthers(except: textEncoder.moduleName)
        }
        if !textEncoder.isLoaded {
            try await textEncoder.load()
        }
        try assertStagedInvariant()
    }

    public func unloadTextEncoderIfStaged() async throws {
        if memoryPolicy != .resident {
            await textEncoder.unload()
            try assertStagedInvariant()
        }
    }

    public func loadDiTExclusive() async throws {
        if memoryPolicy != .resident {
            await unloadOthers(except: dit.moduleName)
        }
        if !dit.isLoaded {
            try await dit.load()
        }
        try assertStagedInvariant()
    }

    public func unloadDiTIfStaged() async throws {
        if memoryPolicy != .resident {
            await dit.unload()
            try assertStagedInvariant()
        }
    }

    public func loadVAEExclusive(mode: VAELoadMode = .full) async throws {
        if memoryPolicy != .resident {
            await unloadOthers(except: vae.moduleName)
        }
        try await vae.load(mode: mode)
        try assertStagedInvariant()
    }

    public func unloadVAEIfStaged() async throws {
        if memoryPolicy != .resident {
            await vae.unload()
            try assertStagedInvariant()
        }
    }

    /// Load DiT (with weights if snapshot set), count params, unload (unless resident).
    public func loadDiTAndCountParameters() async throws -> Int {
        try await loadDiTExclusive()
        let count = dit.parameterLeafCount
        try await unloadDiTIfStaged()
        return count
    }

    public func loadVAEAndCountParameters() async throws -> Int {
        try await loadVAEExclusive()
        let count = vae.parameterLeafCount
        try await unloadVAEIfStaged()
        return count
    }

    public func loadTextEncoderAndCountParameters() async throws -> Int {
        try await loadTextEncoderExclusive()
        let count = textEncoder.parameterLeafCount
        try await unloadTextEncoderIfStaged()
        return count
    }

    public func purge() async {
        await textEncoder.unload()
        await dit.unload()
        await vae.unload()
        // Modules clear on unload when they held weights; always clear residual cache too.
        Memory.clearCache()
        _ = probe.snapshot(label: "purge")
    }

    public func runMemorySelfTest() async throws -> [MemorySample] {
        probe.reset()
        _ = probe.snapshot(label: "start")

        try await loadTextEncoderExclusive()
        _ = probe.snapshot(label: "text_encoder_loaded")
        try await Task.sleep(nanoseconds: 5_000_000)
        try await unloadTextEncoderIfStaged()
        _ = probe.snapshot(label: "after_text_unload")

        try await loadDiTExclusive()
        _ = probe.snapshot(label: "dit_loaded")
        try await Task.sleep(nanoseconds: 5_000_000)
        try await unloadDiTIfStaged()
        _ = probe.snapshot(label: "after_dit_unload")

        try await loadVAEExclusive()
        _ = probe.snapshot(label: "vae_loaded")
        try await Task.sleep(nanoseconds: 5_000_000)
        try await unloadVAEIfStaged()
        _ = probe.snapshot(label: "after_vae_unload")

        try assertStagedInvariant()
        return probe.allSamples()
    }

    private func unloadOthers(except name: String) async {
        if textEncoder.moduleName != name, textEncoder.isLoaded {
            await textEncoder.unload()
        }
        if dit.moduleName != name, dit.isLoaded {
            await dit.unload()
        }
        if vae.moduleName != name, vae.isLoaded {
            await vae.unload()
        }
    }
}
