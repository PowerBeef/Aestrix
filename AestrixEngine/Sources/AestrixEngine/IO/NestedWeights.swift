import Foundation
import MLX
import MLXNN

/// Builds an MLXNN `ModuleParameters` tree (`NestedDictionary<String, MLXArray>`) from flat
/// dotted-path weight keys (e.g. `"encoder.down_blocks.0.resnets.0.conv1.weight"`).
///
/// MLXNN represents array-of-Module children (`resnets`, `down_blocks`, …) as `.array([...])`,
/// so integer-string path segments ("0", "1", …) are converted to positional arrays rather
/// than dictionaries. Named segments stay as dictionaries. The result can be passed to
/// `module.update(parameters:)`.
extension NestedDictionary where Key == String, Element == MLXArray {

    init(flatWeights flat: [String: MLXArray]) {
        var root: [String: NestedItem<String, MLXArray>] = [:]
        for (key, value) in flat {
            let parts = key.split(separator: ".").map(String.init)
            NestedBuilder.insert(&root, parts, .value(value))
        }
        // Convert integer-keyed dictionaries (array children) into `.array`.
        self.init(values: root.mapValues { NestedBuilder.normalize($0) })
    }
}

private enum NestedBuilder {
    static func insert(
        _ dict: inout [String: NestedItem<String, MLXArray>],
        _ parts: [String],
        _ item: NestedItem<String, MLXArray>
    ) {
        if parts.count == 1 {
            dict[parts[0]] = item
            return
        }
        let head = parts[0]
        var node = dict[head] ?? .dictionary([:])
        if case .dictionary(var inner) = node {
            insert(&inner, Array(parts[1...]), item)
            node = .dictionary(inner)
        }
        dict[head] = node
    }

    /// Recursively convert any `.dictionary` whose keys are all integer strings into an
    /// `.array` ordered by those integers (matching MLXNN's array-of-Module layout).
    static func normalize(_ item: NestedItem<String, MLXArray>) -> NestedItem<String, MLXArray> {
        switch item {
        case .dictionary(let d):
            let recursed = d.mapValues { normalize($0) }
            if recursed.isEmpty {
                return .dictionary(recursed)
            }
            let isIndexed = recursed.keys.allSatisfy { Int($0) != nil }
            if isIndexed {
                let ordered = recursed.sorted { (Int($0.key) ?? 0) < (Int($1.key) ?? 0) }
                return .array(ordered.map { $0.value })
            }
            return .dictionary(recursed)
        default:
            return item
        }
    }
}
