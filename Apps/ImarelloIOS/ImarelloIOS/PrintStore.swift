import Foundation
import UIKit
import ImageIO
import ImarelloCore

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
        return side > 0 ? "\(side)" : "print"
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
    /// Metadata for records whose file is currently missing from both
    /// locations. Not rendered, but persisted so nothing is destroyed.
    private var unresolved: [PrintRecord] = []
    private let thumbnails = NSCache<NSString, UIImage>()

    private static let indexVersion = 1

    init() {
        outputsDir = AppCache.directory("outputs")
        let durableRoot = Self.durableRoot()
        printsDir = durableRoot.appendingPathComponent("prints", isDirectory: true)
        indexURL = durableRoot.appendingPathComponent("prints-index.json")
        legacyIndexURL = AppCache.productRoot().appendingPathComponent("prints-index.json")
        let fm = FileManager.default
        try? fm.createDirectory(at: outputsDir, withIntermediateDirectories: true)
        try? fm.createDirectory(at: printsDir, withIntermediateDirectories: true)
        thumbnails.countLimit = 64
        // The store lives for the app's lifetime; the observer is never removed.
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil, queue: .main
        ) { [weak thumbnails] _ in
            thumbnails?.removeAllObjects()
        }
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
        if FileManager.default.fileExists(atPath: durable.path) { return durable }
        return outputsDir.appendingPathComponent(record.fileName)
    }

    func record(
        outputURL: URL, prompt: String, seed: UInt64?, side: Int, mode: String
    ) {
        // Canonical copy into durable storage; the Caches original stays for
        // the harness to pull.
        let durable = printsDir.appendingPathComponent(outputURL.lastPathComponent)
        if durable != outputURL {
            try? FileManager.default.removeItem(at: durable)
            try? FileManager.default.copyItem(at: outputURL, to: durable)
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
        prints.removeAll { $0.id == record.id }
        unresolved.removeAll { $0.id == record.id }
        prints.insert(record, at: 0)
        save()
    }

    func delete(_ record: PrintRecord) {
        try? FileManager.default.removeItem(at: printsDir.appendingPathComponent(record.fileName))
        try? FileManager.default.removeItem(at: outputsDir.appendingPathComponent(record.fileName))
        removeThumbnails(for: record.id)
        prints.removeAll { $0.id == record.id }
        unresolved.removeAll { $0.id == record.id }
        save()
    }

    var latest: PrintRecord? { prints.first }

    /// Downsampled image for the contact sheet (full decode stays in the viewer).
    /// Cached per (record, size bucket) so callers with different sizes don't
    /// alias each other's thumbnails.
    func thumbnail(for record: PrintRecord, maxPixel: CGFloat = 480) -> UIImage? {
        let key = "\(record.id)#\(Int(maxPixel))" as NSString
        if let cached = thumbnails.object(forKey: key) { return cached }
        let source = url(for: record)
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixel * 2,
        ]
        guard
            let src = CGImageSourceCreateWithURL(source as CFURL, nil),
            let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, options as CFDictionary)
        else { return nil }
        let image = UIImage(cgImage: cg)
        thumbnails.setObject(image, forKey: key)
        return image
    }

    private func removeThumbnails(for id: String) {
        for bucket in [480, 512] {
            thumbnails.removeObject(forKey: "\(id)#\(bucket)" as NSString)
        }
    }

    func image(for record: PrintRecord) -> UIImage? {
        UIImage(contentsOfFile: url(for: record).path)
    }

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
        let decoded = decodeIndex(at: indexURL)
            ?? decodeIndex(at: legacyIndexURL)  // pre-durable-storage builds
        guard let decoded else { return }
        let fm = FileManager.default
        var visible: [PrintRecord] = []
        for record in decoded {
            let inDurable = fm.fileExists(atPath: printsDir.appendingPathComponent(record.fileName).path)
            let inOutputs = fm.fileExists(atPath: outputsDir.appendingPathComponent(record.fileName).path)
            if inDurable || inOutputs {
                if !inDurable, inOutputs {
                    // Pull the Caches copy into durable storage.
                    try? fm.copyItem(
                        at: outputsDir.appendingPathComponent(record.fileName),
                        to: printsDir.appendingPathComponent(record.fileName))
                }
                visible.append(record)
            } else {
                // Keep the metadata; the file may reappear (weight resync,
                // partial Caches purge). Never destroy it on the next save.
                unresolved.append(record)
            }
        }
        prints = visible
    }

    private func decodeIndex(at url: URL) -> [PrintRecord]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
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
        try? FileManager.default.moveItem(at: url, to: aside)
        return nil
    }

    private func save() {
        let index = IndexFile(version: Self.indexVersion, prints: prints + unresolved)
        do {
            let data = try JSONEncoder().encode(index)
            try data.write(to: indexURL, options: .atomic)
        } catch {
            print("PrintStore: failed to save index: \(error)")
        }
    }

    /// Adopt PNGs present on disk but absent from the index (pre-history
    /// builds, harness runs from other sessions) with what the filename can
    /// tell us. A file matching an unresolved record restores that record's
    /// full metadata instead.
    private func adoptUnindexedPNGs() {
        let fm = FileManager.default
        let known = Set(prints.map(\.fileName))
        var candidates: [URL] = []
        for dir in [printsDir, outputsDir] {
            if let files = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) {
                candidates.append(contentsOf: files)
            }
        }
        var adopted = false
        var seen = Set<String>()
        for file in candidates
        where file.pathExtension == "png"
            && !known.contains(file.lastPathComponent)
            && !seen.contains(file.lastPathComponent) {
            seen.insert(file.lastPathComponent)
            if let restoredIndex = unresolved.firstIndex(where: { $0.fileName == file.lastPathComponent }) {
                prints.append(unresolved.remove(at: restoredIndex))
                adopted = true
                continue
            }
            let stem = file.deletingPathExtension().lastPathComponent
            let parts = stem.split(separator: "-")
            let mode = parts.first.map(String.init) ?? "t2i"
            let epoch = parts.count > 1 ? TimeInterval(parts[1]) : nil
            let side = Self.pixelSide(of: file)
            prints.append(
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
            prints.sort { $0.createdAt > $1.createdAt }
            save()
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
