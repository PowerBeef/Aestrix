import Foundation
import MLX
import AestrixCore

extension SafetensorsLoader {
    /// Merge all safetensors shards in a directory into one key → array map.
    public static func loadMergedArrays(in directory: URL) throws -> [String: MLXArray] {
        let urls = try shardURLs(in: directory)
        var merged: [String: MLXArray] = [:]
        for url in urls {
            let part = try MLX.loadArrays(url: url)
            for (k, v) in part {
                merged[k] = v
            }
        }
        return merged
    }
}
