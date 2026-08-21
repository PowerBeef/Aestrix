import Foundation
import ImarelloCore

/// Transactional persistence for the frozen device-harness wire contract.
/// Kept separate from UI mutation so fault-injection tests can prove ordering.
@MainActor
final class HarnessPersistence {
    private let root: URL
    private let fileIO: any PrintStoreFileIO

    init(root: URL, fileIO: any PrintStoreFileIO) {
        self.root = root
        self.fileIO = fileIO
    }

    func doneFile(id: String) -> URL {
        root.appendingPathComponent("done", isDirectory: true)
            .appendingPathComponent("\(DeviceHarnessJob.sanitizedID(id)).json")
    }

    func writeResult(_ result: DeviceHarnessResult) throws {
        try ensureDirectories()
        let data = try DeviceHarnessPaths.jsonEncoder.encode(result)
        try fileIO.write(data, to: doneFile(id: result.id))
    }

    /// Commit the result first. The running marker is only removed after the
    /// durable write succeeds, so every interrupted state remains recoverable.
    func finalize(_ result: DeviceHarnessResult, runningURL: URL) throws {
        try writeResult(result)
        try fileIO.removeItem(at: runningURL)
    }

    func hasDurableResult(id: String) -> Bool {
        guard let data = try? fileIO.readData(at: doneFile(id: id)),
              let result = try? DeviceHarnessPaths.jsonDecoder.decode(
                DeviceHarnessResult.self, from: data
              )
        else { return false }
        return result.id == id
    }

    private func ensureDirectories() throws {
        try fileIO.createDirectory(at: root)
        try fileIO.createDirectory(at: root.appendingPathComponent("running", isDirectory: true))
        try fileIO.createDirectory(at: root.appendingPathComponent("done", isDirectory: true))
    }
}
