import Foundation
import MLX
import MLXNN

/// One-shot dump of DiT activation / weight dtypes (R2: M2 bf16-emu check).
///
/// Off by default — must not change numerics or allocate extra when disabled.
public enum ComputeDTypeProbe {
    nonisolated(unsafe) public static var enabled = false
    nonisolated(unsafe) static var lines: [String] = []
    nonisolated(unsafe) static var didDumpBlock = false
    nonisolated(unsafe) static var qkvRecords = 0
    nonisolated(unsafe) static var didDumpAttention = false

    public static func reset() {
        enabled = false
        lines = []
        didDumpBlock = false
        qkvRecords = 0
        didDumpAttention = false
    }

    public static func report() -> String {
        lines.joined(separator: "\n")
    }

    static func record(_ label: String, _ array: MLXArray) {
        guard enabled else { return }
        lines.append("\(label)  dtype=\(dtypeName(array.dtype))  shape=\(array.shape)")
    }

    static func recordNote(_ text: String) {
        guard enabled else { return }
        lines.append(text)
    }

    public static func recordWeights(from model: Flux2Transformer) {
        guard enabled else { return }
        recordLinear("weight.x_embedder", model.xEmbedder)
        recordLinear("weight.context_embedder", model.contextEmbedder)
        recordLinear("weight.double_mod_img", model.doubleStreamModulationImg.linear)
        if let block = model.transformerBlocks.first {
            recordLinear("weight.double0.to_q", block.attn.toQ)
            recordLinear("weight.double0.ff_in", block.ff.linearIn)
        }
    }

    private static func recordLinear(_ label: String, _ linear: Linear) {
        record("\(label).weight", linear.weight)
        if let q = linear as? QuantizedLinear {
            record("\(label).scales", q.scales)
            recordNote("\(label)\tquantized bits=\(q.bits) group=\(q.groupSize) mode=\(q.mode)")
        } else {
            recordNote("\(label)\tnot quantized")
        }
    }

    static func dtypeName(_ dtype: DType) -> String {
        switch dtype {
        case .bool: return "bool"
        case .uint8: return "uint8"
        case .uint16: return "uint16"
        case .uint32: return "uint32"
        case .uint64: return "uint64"
        case .int8: return "int8"
        case .int16: return "int16"
        case .int32: return "int32"
        case .int64: return "int64"
        case .float16: return "float16"
        case .float32: return "float32"
        case .bfloat16: return "bfloat16"
        case .complex64: return "complex64"
        default: return String(describing: dtype)
        }
    }
}
