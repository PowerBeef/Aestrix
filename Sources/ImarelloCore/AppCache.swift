import Foundation

/// On-disk product cache. New writes go under `Imarello/`.
/// Existing `Aestrix/` snapshots, embeds, and CLIP models are still read.
public enum AppCache {
    public static let productFolder = "Imarello"
    /// Pre-rename cache folder. Do not write here.
    public static let legacyFolder = "Aestrix"

    public static func userCaches() -> URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
    }

    public static func productRoot() -> URL {
        userCaches().appendingPathComponent(productFolder, isDirectory: true)
    }

    public static func legacyRoot() -> URL {
        userCaches().appendingPathComponent(legacyFolder, isDirectory: true)
    }

    /// Preferred directory for new files (`Imarello/<name>`).
    public static func directory(_ name: String) -> URL {
        productRoot().appendingPathComponent(name, isDirectory: true)
    }

    /// First existing `Imarello/<name>` or `Aestrix/<name>`, else the Imarello path.
    public static func resolvedDirectory(_ name: String) -> URL {
        let neu = directory(name)
        if directoryExists(neu) { return neu }
        let old = legacyRoot().appendingPathComponent(name, isDirectory: true)
        if directoryExists(old) { return old }
        return neu
    }

    /// First existing file/dir among Imarello then Aestrix, else the Imarello candidate.
    public static func resolvedItem(under name: String, item: String) -> URL {
        let neu = directory(name).appendingPathComponent(item)
        if FileManager.default.fileExists(atPath: neu.path) { return neu }
        let old = legacyRoot().appendingPathComponent(name, isDirectory: true)
            .appendingPathComponent(item)
        if FileManager.default.fileExists(atPath: old.path) { return old }
        return neu
    }

    public static func directoryExists(_ url: URL) -> Bool {
        var isDir: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir)
            && isDir.boolValue
    }
}
