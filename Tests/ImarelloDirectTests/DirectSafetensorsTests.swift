import Foundation
import Testing
@testable import ImarelloDirect

@Suite("Strict Direct safetensors index")
struct DirectSafetensorsTests {
    @Test("valid descriptors resolve to absolute file ranges")
    func validIndex() throws {
        let file = makeFile(
            header: #"{"a":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"U32","shape":[1],"data_offsets":[4,8]}}"#,
            payloadBytes: 8)
        let index = try SafetensorIndex.parse(
            data: file,
            allowedTensorNames: ["a", "b"],
            requiredTensorNames: ["a", "b"])
        #expect(index.tensors.map(\.name) == ["a", "b"])
        #expect(index.tensors[0].byteRange == index.payloadOffset ..< index.payloadOffset + 4)
        #expect(index.tensors[1].byteRange == index.payloadOffset + 4 ..< index.payloadOffset + 8)
    }

    @Test("truncated prefix and header fail before JSON decoding")
    func truncation() {
        #expect(throws: SafetensorError.truncatedPrefix) {
            try SafetensorIndex.parse(data: Data([0, 1, 2]))
        }
        var prefix = littleEndianBytes(64)
        prefix.append(contentsOf: [0x7B, 0x7D])
        #expect(throws: SafetensorError.truncatedHeader(expectedEnd: 72, fileSize: 10)) {
            try SafetensorIndex.parse(data: Data(prefix))
        }
    }

    @Test("duplicate tensor names are rejected before dictionary collapse")
    func duplicateName() {
        let file = makeFile(
            header: #"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#,
            payloadBytes: 2)
        #expect(throws: SafetensorError.duplicateTensor("a")) {
            try SafetensorIndex.parse(data: file)
        }
    }

    @Test("shape byte count, bounds, and overlap are independently validated")
    func rangesAndShapes() {
        let mismatch = makeFile(
            header: #"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
            payloadBytes: 4)
        #expect(throws: SafetensorError.byteCountMismatch(
            name: "a", expected: 8, actual: 4)
        ) {
            try SafetensorIndex.parse(data: mismatch)
        }

        let outOfBounds = makeFile(
            header: #"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#,
            payloadBytes: 1)
        #expect(throws: SafetensorError.outOfBounds("a")) {
            try SafetensorIndex.parse(data: outOfBounds)
        }

        let overlap = makeFile(
            header: #"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}"#,
            payloadBytes: 3)
        #expect(throws: SafetensorError.overlappingTensors("a", "b")) {
            try SafetensorIndex.parse(data: overlap)
        }
    }

    @Test("allow and require manifests fail closed")
    func nameManifest() {
        let file = makeFile(
            header: #"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
            payloadBytes: 1)
        #expect(throws: SafetensorError.unexpectedTensor("a")) {
            try SafetensorIndex.parse(data: file, allowedTensorNames: ["b"])
        }
        #expect(throws: SafetensorError.missingTensor("b")) {
            try SafetensorIndex.parse(data: file, requiredTensorNames: ["a", "b"])
        }
    }

    private func makeFile(header: String, payloadBytes: Int) -> Data {
        var headerData = Data(header.utf8)
        while !headerData.count.isMultiple(of: 8) { headerData.append(0x20) }
        var bytes = Data(littleEndianBytes(UInt64(headerData.count)))
        bytes.append(headerData)
        bytes.append(Data(repeating: 0, count: payloadBytes))
        return bytes
    }

    private func littleEndianBytes<T: FixedWidthInteger>(_ value: T) -> [UInt8] {
        let little = value.littleEndian
        return withUnsafeBytes(of: little) { Array($0) }
    }
}
