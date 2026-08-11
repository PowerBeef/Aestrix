import Foundation
import AestrixCore

/// Load safetensors shards from a module directory (uses index.json when present).
public enum SafetensorsLoader {
    public static func shardFileNames(in directory: URL) throws -> [String] {
        let fm = FileManager.default
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if fm.fileExists(atPath: indexURL.path),
           let data = try? Data(contentsOf: indexURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let map = json["weight_map"] as? [String: String]
        {
            return Array(Set(map.values)).sorted()
        }
        let contents = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let names = contents
            .filter { $0.pathExtension == "safetensors" }
            .map(\.lastPathComponent)
            .sorted()
        guard !names.isEmpty else {
            throw AestrixError.weightsNotFound(modelID: directory.lastPathComponent, path: directory.path)
        }
        return names
    }

    public static func shardURLs(in directory: URL) throws -> [URL] {
        try shardFileNames(in: directory).map { directory.appendingPathComponent($0) }
    }
}
