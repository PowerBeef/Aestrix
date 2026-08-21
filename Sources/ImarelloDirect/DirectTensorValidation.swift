import MLX
import Metal

enum DirectTensorValidation {
    static let groupSize = 64
    static let valuesPerPackedWord = 8

    static func requireShape(_ array: MLXArray, _ expected: [Int], name: String) throws {
        guard array.shape == expected else {
            throw DirectQmmSpike.SpikeError.invalidTensor(
                "\(name) expected shape \(expected), got \(array.shape)")
        }
    }

    static func requireQuantized(
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        n: Int,
        k: Int,
        name: String
    ) throws {
        guard k.isMultiple(of: groupSize), k.isMultiple(of: valuesPerPackedWord) else {
            throw DirectQmmSpike.SpikeError.invalidTensor(
                "\(name) unsupported quantization dimensions n=\(n), k=\(k)")
        }
        try requireShape(weight, [n, k / valuesPerPackedWord], name: "\(name).weight")
        try requireShape(scales, [n, k / groupSize], name: "\(name).scales")
        try requireShape(biases, [n, k / groupSize], name: "\(name).biases")
    }

    static func requireBuffer(_ buffer: MTLBuffer, floatCount: Int, name: String) throws {
        let expectedBytes = floatCount * MemoryLayout<Float>.stride
        guard buffer.length == expectedBytes else {
            throw DirectQmmSpike.SpikeError.invalidTensor(
                "\(name) expected \(expectedBytes) bytes, got \(buffer.length)")
        }
    }
}
