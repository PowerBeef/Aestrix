import Foundation

/// Deterministic engine-owned accounting. Device-wide allocation remains a
/// separate observation because MLX, drivers, and externally retained bridges
/// are not owned by this ledger.
public struct DirectMemoryLedger: Codable, Sendable, Equatable {
    public private(set) var scratchAllocatedBytes = 0
    public private(set) var scratchPeakBytes = 0
    public private(set) var persistentWeightBytes = 0
    public private(set) var conditioningBytes = 0
    public private(set) var bridgeAllocatedBytes = 0
    public private(set) var externallyOwnedMappedBytes = 0
    public private(set) var cumulativeUploadBytes = 0

    public var liveEngineOwnedBytes: Int {
        scratchAllocatedBytes + persistentWeightBytes + conditioningBytes
    }

    public var totalAllocatedBytes: Int {
        liveEngineOwnedBytes + bridgeAllocatedBytes
    }

    mutating func recordScratch(bytes: Int) {
        scratchAllocatedBytes += bytes
        scratchPeakBytes = max(scratchPeakBytes, scratchAllocatedBytes)
    }

    mutating func recordPersistentUpload(bytes: Int) {
        persistentWeightBytes += bytes
        cumulativeUploadBytes += bytes
    }

    mutating func recordBridgeUpload(bytes: Int) {
        bridgeAllocatedBytes += bytes
        cumulativeUploadBytes += bytes
    }

    mutating func recordConditioningUpload(bytes: Int) {
        cumulativeUploadBytes += bytes
    }

    mutating func replaceConditioning(bytes: Int) {
        conditioningBytes = bytes
    }

    mutating func replaceExternallyOwnedMappings(bytes: Int) {
        externallyOwnedMappedBytes = bytes
    }
}
