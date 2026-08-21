import Foundation
import ImageIO
import ImarelloCore

enum PrintStoreError: LocalizedError {
    case invalidPNG(String)

    var errorDescription: String? {
        switch self {
        case .invalidPNG(let detail):
            return "The generated image could not be saved: \(detail)"
        }
    }
}

/// Narrow filesystem seam used by the iOS durability tests to inject failures
/// at copy, rename, delete, and index-write boundaries.
@MainActor
protocol PrintStoreFileIO: AnyObject {
    func fileExists(at url: URL) -> Bool
    func createDirectory(at url: URL) throws
    func contentsOfDirectory(at url: URL) throws -> [URL]
    func copyItem(at source: URL, to destination: URL) throws
    func moveItem(at source: URL, to destination: URL) throws
    func removeItem(at url: URL) throws
    func readData(at url: URL) throws -> Data
    func write(_ data: Data, to url: URL) throws
}

@MainActor
final class SystemPrintStoreFileIO: PrintStoreFileIO {
    private let fm = FileManager.default

    func fileExists(at url: URL) -> Bool { fm.fileExists(atPath: url.path) }
    func createDirectory(at url: URL) throws {
        try fm.createDirectory(at: url, withIntermediateDirectories: true)
    }
    func contentsOfDirectory(at url: URL) throws -> [URL] {
        try fm.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
    }
    func copyItem(at source: URL, to destination: URL) throws {
        try fm.copyItem(at: source, to: destination)
    }
    func moveItem(at source: URL, to destination: URL) throws {
        try fm.moveItem(at: source, to: destination)
    }
    func removeItem(at url: URL) throws { try fm.removeItem(at: url) }
    func readData(at url: URL) throws -> Data { try Data(contentsOf: url) }
    func write(_ data: Data, to url: URL) throws { try data.write(to: url, options: .atomic) }
}

/// One finished print. Records and canonical pixels live in Application
/// Support (durable); the harness copy in `Caches/Imarello/outputs/` stays
/// where `Scripts/ios-device-harness.sh` expects it (frozen contract).
struct PrintRecord: Codable, Identifiable, Equatable {
    let id: String
    let fileName: String
    let prompt: String
    let seed: UInt64?
    let side: Int
    /// "t2i" | "i2i"
    let mode: String
    let createdAt: Date

    /// Caption contract: `{side} · seed {n}` for the print that actually ran.
    var caption: String {
        if let seed { return "\(side) · seed \(seed)" }
        return side > 0 ? "\(side)" : "image"
    }
}

/// Persistent history of prints.
///
/// Durability: `Caches` is OS-purgeable, so the index and a canonical copy of
/// every print live under Application Support. The Caches `outputs/` copy is
/// kept for the frozen device-harness contract and treated as expendable.
///
/// Robustness: the index file is versioned, decoded per-element (one bad row
/// never drops the rest), and an undecodable index is renamed aside — never
/// silently overwritten. Records whose PNG is momentarily missing keep their
/// metadata and are restored if the file reappears.
@MainActor
@Observable
final class PrintStore {
    private(set) var prints: [PrintRecord] = []

    private let indexURL: URL
    private let printsDir: URL
    private let outputsDir: URL
    private let legacyIndexURL: URL
    private let fileIO: any PrintStoreFileIO
    /// Metadata for records whose file is currently missing from both
    /// locations. Not rendered, but persisted so nothing is destroyed.
    private var unresolved: [PrintRecord] = []
    private static let indexVersion = 1

    convenience init() {
        self.init(
            durableRoot: Self.durableRoot(),
            outputsDirectory: AppCache.directory("outputs"),
            legacyIndexURL: AppCache.productRoot().appendingPathComponent("prints-index.json"),
            fileIO: SystemPrintStoreFileIO()
        )
    }

    init(
        durableRoot: URL,
        outputsDirectory: URL,
        legacyIndexURL: URL,
        fileIO: any PrintStoreFileIO
    ) {
        self.fileIO = fileIO
        outputsDir = outputsDirectory
        printsDir = durableRoot.appendingPathComponent("prints", isDirectory: true)
        indexURL = durableRoot.appendingPathComponent("prints-index.json")
        self.legacyIndexURL = legacyIndexURL
        try? fileIO.createDirectory(at: outputsDir)
        try? fileIO.createDirectory(at: printsDir)
        load()
        adoptUnindexedPNGs()
    }

    private static func durableRoot() -> URL {
        let base = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? AppCache.productRoot()
        let root = base.appendingPathComponent("Imarello", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    func url(for record: PrintRecord) -> URL {
        let durable = printsDir.appendingPathComponent(record.fileName)
        if fileIO.fileExists(at: durable) { return durable }
        return outputsDir.appendingPathComponent(record.fileName)
    }

    @discardableResult
    func record(
        outputURL: URL, prompt: String, seed: UInt64?, side: Int, mode: String
    ) throws -> PrintRecord {
        try Self.validatePNG(at: outputURL, expectedSide: side)
        let durable = printsDir.appendingPathComponent(outputURL.lastPathComponent)
        let transactionID = UUID().uuidString
        let incoming = printsDir.appendingPathComponent(".incoming-\(transactionID).png")
        let backup = printsDir.appendingPathComponent(".backup-\(transactionID).png")
        var hadPrevious = false

        if durable != outputURL {
            try fileIO.copyItem(at: outputURL, to: incoming)
            do {
                try Self.validatePNG(at: incoming, expectedSide: side)
                if fileIO.fileExists(at: durable) {
                    try fileIO.moveItem(at: durable, to: backup)
                    hadPrevious = true
                }
                try fileIO.moveItem(at: incoming, to: durable)
            } catch {
                try? fileIO.removeItem(at: incoming)
                if hadPrevious, !fileIO.fileExists(at: durable) {
                    try? fileIO.moveItem(at: backup, to: durable)
                }
                throw error
            }
        }
        let record = PrintRecord(
            id: outputURL.deletingPathExtension().lastPathComponent,
            fileName: outputURL.lastPathComponent,
            prompt: prompt,
            seed: seed,
            side: side,
            mode: mode,
            createdAt: Date()
        )
        var proposedPrints = prints.filter { $0.id != record.id }
        let proposedUnresolved = unresolved.filter { $0.id != record.id }
        proposedPrints.insert(record, at: 0)
        do {
            try save(prints: proposedPrints, unresolved: proposedUnresolved)
        } catch {
            if durable != outputURL {
                try? fileIO.removeItem(at: durable)
                if hadPrevious { try? fileIO.moveItem(at: backup, to: durable) }
            }
            throw error
        }
        if hadPrevious { try? fileIO.removeItem(at: backup) }
        prints = proposedPrints
        unresolved = proposedUnresolved
        return record
    }

    func delete(_ record: PrintRecord) throws {
        let transactionID = UUID().uuidString
        let sources = [
            printsDir.appendingPathComponent(record.fileName),
            outputsDir.appendingPathComponent(record.fileName),
        ]
        var staged: [(original: URL, staged: URL)] = []
        do {
            for (index, source) in sources.enumerated() where fileIO.fileExists(at: source) {
                let destination = source.deletingLastPathComponent()
                    .appendingPathComponent(".deleted-\(transactionID)-\(index)")
                try fileIO.moveItem(at: source, to: destination)
                staged.append((source, destination))
            }
        } catch {
            restore(staged)
            throw error
        }

        let proposedPrints = prints.filter { $0.id != record.id }
        let proposedUnresolved = unresolved.filter { $0.id != record.id }
        do {
            try save(prints: proposedPrints, unresolved: proposedUnresolved)
        } catch {
            restore(staged)
            throw error
        }
        prints = proposedPrints
        unresolved = proposedUnresolved
        for item in staged { try? fileIO.removeItem(at: item.staged) }
    }

    var latest: PrintRecord? { prints.first }

    // MARK: - persistence

    /// Versioned envelope; the pre-versioned format was a bare array.
    private struct IndexFile: Codable {
        var version: Int
        var prints: [PrintRecord]
    }

    /// Decodes each element independently so one bad row never drops the rest.
    private struct LossyRecord: Decodable {
        let value: PrintRecord?
        init(from decoder: Decoder) throws { value = try? PrintRecord(from: decoder) }
    }

    private struct LossyIndexFile: Decodable {
        var version: Int
        var prints: [LossyRecord]
    }

    private func load() {
        let current = decodeIndex(at: indexURL)
        let decoded = current ?? decodeIndex(at: legacyIndexURL)
        guard let decoded else { return }
        var visible: [PrintRecord] = []
        var missing: [PrintRecord] = []
        for record in decoded {
            let durable = printsDir.appendingPathComponent(record.fileName)
            let output = outputsDir.appendingPathComponent(record.fileName)
            let inDurable = fileIO.fileExists(at: durable)
            let inOutputs = fileIO.fileExists(at: output)
            if inDurable || inOutputs {
                if !inDurable, inOutputs {
                    // Pull the Caches copy into durable storage.
                    try? fileIO.copyItem(at: output, to: durable)
                }
                visible.append(record)
            } else {
                // Keep the metadata; the file may reappear (weight resync,
                // partial Caches purge). Never destroy it on the next save.
                missing.append(record)
            }
        }
        prints = visible
        unresolved = missing
        // A successfully decoded legacy index is migrated immediately while
        // every original metadata field is still available.
        if current == nil {
            try? save(prints: visible, unresolved: missing)
        }
    }

    private func decodeIndex(at url: URL) -> [PrintRecord]? {
        guard let data = try? fileIO.readData(at: url) else { return nil }
        let decoder = JSONDecoder()
        if let envelope = try? decoder.decode(LossyIndexFile.self, from: data) {
            return envelope.prints.compactMap(\.value)
        }
        if let bare = try? decoder.decode([LossyRecord].self, from: data) {
            return bare.compactMap(\.value)
        }
        // Undecodable: preserve the file for recovery instead of letting a
        // later save() overwrite it with adopted, metadata-less records.
        let stamp = Int(Date().timeIntervalSince1970)
        let aside = url.deletingLastPathComponent()
            .appendingPathComponent("prints-index.bad-\(stamp).json")
        try? fileIO.moveItem(at: url, to: aside)
        return nil
    }

    private func save(prints: [PrintRecord], unresolved: [PrintRecord]) throws {
        let index = IndexFile(version: Self.indexVersion, prints: prints + unresolved)
        let data = try JSONEncoder().encode(index)
        try fileIO.write(data, to: indexURL)
    }

    /// Adopt PNGs present on disk but absent from the index (pre-history
    /// builds, harness runs from other sessions) with what the filename can
    /// tell us. A file matching an unresolved record restores that record's
    /// full metadata instead.
    private func adoptUnindexedPNGs() {
        let known = Set(prints.map(\.fileName))
        var candidates: [URL] = []
        for dir in [printsDir, outputsDir] {
            if let files = try? fileIO.contentsOfDirectory(at: dir) {
                candidates.append(contentsOf: files)
            }
        }
        var proposedPrints = prints
        var proposedUnresolved = unresolved
        var adopted = false
        var seen = Set<String>()
        for file in candidates
        where file.pathExtension == "png"
            && !known.contains(file.lastPathComponent)
            && !seen.contains(file.lastPathComponent) {
            seen.insert(file.lastPathComponent)
            if let restoredIndex = proposedUnresolved.firstIndex(
                where: { $0.fileName == file.lastPathComponent }
            ) {
                proposedPrints.append(proposedUnresolved.remove(at: restoredIndex))
                adopted = true
                continue
            }
            let stem = file.deletingPathExtension().lastPathComponent
            let parts = stem.split(separator: "-")
            let mode = parts.first.map(String.init) ?? "t2i"
            let epoch = parts.count > 1 ? TimeInterval(parts[1]) : nil
            let side = Self.pixelSide(of: file)
            proposedPrints.append(
                PrintRecord(
                    id: stem,
                    fileName: file.lastPathComponent,
                    prompt: "",
                    seed: nil,
                    side: side,
                    mode: mode == "i2i" ? "i2i" : "t2i",
                    createdAt: epoch.map(Date.init(timeIntervalSince1970:)) ?? Date.distantPast
                ))
            adopted = true
        }
        if adopted {
            proposedPrints.sort { $0.createdAt > $1.createdAt }
            do {
                try save(prints: proposedPrints, unresolved: proposedUnresolved)
                prints = proposedPrints
                unresolved = proposedUnresolved
            } catch {
                print("PrintStore: failed to adopt recovered PNG metadata: \(error)")
            }
        }
    }

    private func restore(_ staged: [(original: URL, staged: URL)]) {
        for item in staged.reversed() where fileIO.fileExists(at: item.staged) {
            try? fileIO.moveItem(at: item.staged, to: item.original)
        }
    }

    private static func validatePNG(at url: URL, expectedSide: Int) throws {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              CGImageSourceGetCount(source) == 1,
              CGImageSourceGetStatusAtIndex(source, 0) == .statusComplete,
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int,
              width > 0,
              height > 0
        else {
            throw PrintStoreError.invalidPNG("the PNG is incomplete or unreadable")
        }
        if expectedSide > 0, width != expectedSide || height != expectedSide {
            throw PrintStoreError.invalidPNG(
                "expected \(expectedSide)×\(expectedSide) pixels, got \(width)×\(height)"
            )
        }
    }

    private static func pixelSide(of url: URL) -> Int {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
            let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
            let width = props[kCGImagePropertyPixelWidth] as? Int
        else { return 0 }
        return width
    }
}
