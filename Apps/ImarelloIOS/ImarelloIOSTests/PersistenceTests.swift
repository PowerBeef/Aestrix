import Foundation
import Testing
import UIKit
import ImarelloCore
@testable import Imarello

@MainActor
@Suite("Print and harness durability", .serialized)
struct PersistenceTests {
    @Test("print index publishes only after a validated canonical copy")
    func transactionalRecord() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }
        let store = fixture.makeStore()
        let output = try fixture.png(named: "t2i-1.png", side: 4)

        let record = try store.record(
            outputURL: output, prompt: "preserved prompt", seed: 7, side: 4, mode: "t2i"
        )

        #expect(store.latest == record)
        #expect(FileManager.default.fileExists(
            atPath: fixture.durable.appendingPathComponent("prints/t2i-1.png").path
        ))
        let index = try String(
            contentsOf: fixture.durable.appendingPathComponent("prints-index.json"),
            encoding: .utf8
        )
        #expect(index.contains("preserved prompt"))
    }

    @Test("copy failure never creates an index row")
    func copyFailureRollsBack() throws {
        let fixture = try Fixture(failure: .copy)
        defer { fixture.cleanup() }
        let store = fixture.makeStore()
        let output = try fixture.png(named: "t2i-copy.png", side: 4)

        #expect(throws: (any Error).self) {
            try store.record(outputURL: output, prompt: "p", seed: 1, side: 4, mode: "t2i")
        }
        #expect(store.prints.isEmpty)
        #expect(!FileManager.default.fileExists(
            atPath: fixture.durable.appendingPathComponent("prints-index.json").path
        ))
    }

    @Test("canonical replacement failure preserves the source and index")
    func replacementFailureRollsBack() throws {
        let fixture = try Fixture(failure: .move)
        defer { fixture.cleanup() }
        let store = fixture.makeStore()
        let output = try fixture.png(named: "t2i-move.png", side: 4)

        #expect(throws: (any Error).self) {
            try store.record(
                outputURL: output, prompt: "p", seed: 1, side: 4, mode: "t2i"
            )
        }
        #expect(store.prints.isEmpty)
        #expect(FileManager.default.fileExists(atPath: output.path))
        #expect(!FileManager.default.fileExists(
            atPath: fixture.durable.appendingPathComponent("prints-index.json").path
        ))
    }

    @Test("delete staging failure preserves the print and metadata")
    func deleteMoveFailureRollsBack() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }
        let store = fixture.makeStore()
        let output = try fixture.png(named: "t2i-delete.png", side: 4)
        let record = try store.record(
            outputURL: output, prompt: "metadata", seed: 2, side: 4, mode: "t2i"
        )

        fixture.io.failure = .move
        #expect(throws: (any Error).self) { try store.delete(record) }
        #expect(store.latest == record)
        #expect(FileManager.default.fileExists(atPath: store.url(for: record).path))
    }

    @Test("index failure restores files and in-memory state")
    func indexFailureRollsBackRecordAndDelete() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }
        let store = fixture.makeStore()
        let output = try fixture.png(named: "t2i-index.png", side: 4)
        let record = try store.record(
            outputURL: output, prompt: "p", seed: 2, side: 4, mode: "t2i"
        )

        fixture.io.failure = .write
        #expect(throws: (any Error).self) { try store.delete(record) }
        #expect(store.latest == record)
        #expect(FileManager.default.fileExists(atPath: output.path))
        #expect(FileManager.default.fileExists(atPath: store.url(for: record).path))
    }

    @Test("legacy index migrates without losing metadata")
    func legacyMigration() throws {
        let fixture = try Fixture()
        defer { fixture.cleanup() }
        let output = try fixture.png(named: "t2i-legacy.png", side: 4)
        let record = PrintRecord(
            id: "t2i-legacy", fileName: output.lastPathComponent,
            prompt: "legacy metadata", seed: 99, side: 4, mode: "t2i",
            createdAt: Date(timeIntervalSince1970: 1)
        )
        try JSONEncoder().encode([record]).write(to: fixture.legacy, options: .atomic)

        let store = fixture.makeStore()

        #expect(store.latest?.prompt == "legacy metadata")
        #expect(FileManager.default.fileExists(
            atPath: fixture.durable.appendingPathComponent("prints-index.json").path
        ))
    }

    @Test("failed result write retains running marker")
    func harnessWriteFailureRetainsRecovery() throws {
        let fixture = try Fixture(failure: .write)
        defer { fixture.cleanup() }
        let running = fixture.root.appendingPathComponent("jobs/running/job.json")
        try FileManager.default.createDirectory(
            at: running.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try Data("job".utf8).write(to: running)
        let persistence = HarnessPersistence(
            root: fixture.root.appendingPathComponent("jobs"), fileIO: fixture.io
        )
        let result = DeviceHarnessResult(
            id: "job", status: .failed, width: 512, height: 512, seed: 1
        )

        #expect(throws: (any Error).self) {
            try persistence.finalize(result, runningURL: running)
        }
        #expect(FileManager.default.fileExists(atPath: running.path))
        #expect(!persistence.hasDurableResult(id: "job"))
    }

    @Test("committed result survives interrupted marker cleanup")
    func harnessCleanupFailureKeepsDurableResult() throws {
        let fixture = try Fixture(failure: .remove)
        defer { fixture.cleanup() }
        let running = fixture.root.appendingPathComponent("jobs/running/job.json")
        try FileManager.default.createDirectory(
            at: running.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try Data("job".utf8).write(to: running)
        let persistence = HarnessPersistence(
            root: fixture.root.appendingPathComponent("jobs"), fileIO: fixture.io
        )
        let result = DeviceHarnessResult(
            id: "job", status: .ok, width: 512, height: 512, seed: 1
        )

        #expect(throws: (any Error).self) {
            try persistence.finalize(result, runningURL: running)
        }
        #expect(FileManager.default.fileExists(atPath: running.path))
        #expect(persistence.hasDurableResult(id: "job"))
    }
}

@MainActor
private final class FaultingFileIO: PrintStoreFileIO {
    enum Operation { case copy, move, remove, write }
    var failure: Operation?
    private let system = SystemPrintStoreFileIO()

    init(failure: Operation? = nil) { self.failure = failure }

    private func fail(_ operation: Operation) throws {
        guard failure == operation else { return }
        failure = nil
        throw CocoaError(.fileWriteUnknown)
    }

    func fileExists(at url: URL) -> Bool { system.fileExists(at: url) }
    func createDirectory(at url: URL) throws { try system.createDirectory(at: url) }
    func contentsOfDirectory(at url: URL) throws -> [URL] {
        try system.contentsOfDirectory(at: url)
    }
    func copyItem(at source: URL, to destination: URL) throws {
        try fail(.copy)
        try system.copyItem(at: source, to: destination)
    }
    func moveItem(at source: URL, to destination: URL) throws {
        try fail(.move)
        try system.moveItem(at: source, to: destination)
    }
    func removeItem(at url: URL) throws {
        try fail(.remove)
        try system.removeItem(at: url)
    }
    func readData(at url: URL) throws -> Data { try system.readData(at: url) }
    func write(_ data: Data, to url: URL) throws {
        try fail(.write)
        try system.write(data, to: url)
    }
}

@MainActor
private final class Fixture {
    let root: URL
    let durable: URL
    let outputs: URL
    let legacy: URL
    let io: FaultingFileIO

    init(failure: FaultingFileIO.Operation? = nil) throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ImarelloIOSTests-\(UUID().uuidString)", isDirectory: true)
        durable = root.appendingPathComponent("durable", isDirectory: true)
        outputs = root.appendingPathComponent("outputs", isDirectory: true)
        legacy = root.appendingPathComponent("legacy-index.json")
        io = FaultingFileIO(failure: failure)
        try FileManager.default.createDirectory(at: outputs, withIntermediateDirectories: true)
    }

    func makeStore() -> PrintStore {
        PrintStore(
            durableRoot: durable,
            outputsDirectory: outputs,
            legacyIndexURL: legacy,
            fileIO: io
        )
    }

    func png(named name: String, side: Int) throws -> URL {
        let url = outputs.appendingPathComponent(name)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(
            size: CGSize(width: side, height: side), format: format
        )
        let data = renderer.pngData { context in
            UIColor.systemRed.setFill()
            context.fill(CGRect(x: 0, y: 0, width: side, height: side))
        }
        try data.write(to: url, options: .atomic)
        return url
    }

    func cleanup() { try? FileManager.default.removeItem(at: root) }
}
