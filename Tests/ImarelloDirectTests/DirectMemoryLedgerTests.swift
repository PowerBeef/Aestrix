import Testing
@testable import ImarelloDirect

@Suite("Direct memory ledger")
struct DirectMemoryLedgerTests {
    @Test("live memory, bridges, mappings, and cumulative uploads remain distinct")
    func categories() {
        var ledger = DirectMemoryLedger()
        ledger.recordScratch(bytes: 100)
        ledger.recordPersistentUpload(bytes: 40)
        ledger.recordConditioningUpload(bytes: 30)
        ledger.replaceConditioning(bytes: 20)
        ledger.recordBridgeUpload(bytes: 50)
        ledger.replaceExternallyOwnedMappings(bytes: 70)

        #expect(ledger.scratchPeakBytes == 100)
        #expect(ledger.liveEngineOwnedBytes == 160)
        #expect(ledger.totalAllocatedBytes == 210)
        #expect(ledger.cumulativeUploadBytes == 120)
        #expect(ledger.externallyOwnedMappedBytes == 70)
    }
}
