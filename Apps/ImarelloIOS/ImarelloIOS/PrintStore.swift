import Foundation
import UIKit
import ImageIO
import ImarelloCore

/// One finished print. Records live in `Caches/Imarello/prints-index.json`;
/// pixels live in `Caches/Imarello/outputs/` (the frozen harness location).
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

/// Persistent history of prints. Index is a small JSON sidecar; unknown PNGs
/// already in the outputs directory (pre-history builds, harness runs from
/// other sessions) are adopted with what the filename can tell us.
@MainActor
@Observable
final class PrintStore {
    private(set) var prints: [PrintRecord] = []

    private let indexURL: URL
    private let outputsDir: URL
    private var thumbnails: [String: UIImage] = [:]

    init() {
        outputsDir = AppCache.directory("outputs")
        try? FileManager.default.createDirectory(at: outputsDir, withIntermediateDirectories: true)
        indexURL = AppCache.productRoot().appendingPathComponent("prints-index.json")
        load()
        adoptUnindexedPNGs()
    }

    func url(for record: PrintRecord) -> URL {
        outputsDir.appendingPathComponent(record.fileName)
    }

    func record(
        outputURL: URL, prompt: String, seed: UInt64?, side: Int, mode: String
    ) {
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
        prints.insert(record, at: 0)
        save()
    }

    func delete(_ record: PrintRecord) {
        try? FileManager.default.removeItem(at: url(for: record))
        thumbnails[record.id] = nil
        prints.removeAll { $0.id == record.id }
        save()
    }

    var latest: PrintRecord? { prints.first }

    /// Downsampled image for the contact sheet (full decode stays in the viewer).
    func thumbnail(for record: PrintRecord, maxPixel: CGFloat = 480) -> UIImage? {
        if let cached = thumbnails[record.id] { return cached }
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
        thumbnails[record.id] = image
        return image
    }

    func image(for record: PrintRecord) -> UIImage? {
        UIImage(contentsOfFile: url(for: record).path)
    }

    // MARK: - persistence

    private func load() {
        guard let data = try? Data(contentsOf: indexURL),
            let decoded = try? JSONDecoder().decode([PrintRecord].self, from: data)
        else { return }
        // Drop records whose file vanished (container swaps, manual cleanup).
        prints = decoded.filter {
            FileManager.default.fileExists(atPath: outputsDir.appendingPathComponent($0.fileName).path)
        }
    }

    private func save() {
        guard let data = try? JSONEncoder().encode(prints) else { return }
        try? data.write(to: indexURL, options: .atomic)
    }

    /// First-launch migration: PNGs already on disk become records with the
    /// metadata the filename carries (`t2i-<epoch>.png` / `i2i-<epoch>.png`).
    private func adoptUnindexedPNGs() {
        let known = Set(prints.map(\.fileName))
        guard
            let files = try? FileManager.default.contentsOfDirectory(
                at: outputsDir, includingPropertiesForKeys: nil)
        else { return }
        var adopted = false
        for file in files where file.pathExtension == "png" && !known.contains(file.lastPathComponent) {
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
