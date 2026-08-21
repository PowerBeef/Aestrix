import Foundation
import Testing
import ImarelloPlan
@testable import ImarelloDirect

@Suite("Mapped Direct safetensors")
struct MappedSafetensorsTests {
    @Test("page-rounded segment retains exact tensor subranges")
    func mapsSegment() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mapped-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let header: [String: Any] = [
            "a": ["dtype": "U8", "shape": [3], "data_offsets": [0, 3]],
            "b": ["dtype": "U8", "shape": [4], "data_offsets": [3, 7]],
        ]
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while !json.count.isMultiple(of: 8) { json.append(0x20) }
        var length = UInt64(json.count).littleEndian
        var file = Data(bytes: &length, count: 8)
        file.append(json)
        file.append(contentsOf: [1, 2, 3, 4, 5, 6, 7])
        try file.write(to: url)

        let mapped = try MappedSafetensorFile(url: url)
        let segment = try mapped.map(tensorNames: ["a", "b"])
        let a = try #require(segment.withUnsafeTensorBytes(named: "a") { Data($0) })
        let b = try #require(segment.withUnsafeTensorBytes(named: "b") { Data($0) })
        #expect(a == Data([1, 2, 3]))
        #expect(b == Data([4, 5, 6, 7]))
        #expect(segment.mappedFileRange.lowerBound.isMultiple(of: Int(getpagesize())))
        #expect(segment.mappedFileRange.upperBound.isMultiple(of: Int(getpagesize())))
    }

    @Test("transform manifest validates dtype, shape, and affine int4 metadata")
    func transformManifest() throws {
        let data = makeSingleTensor()
        let index = try SafetensorIndex.parse(data: data)
        let valid = DirectTensorTransformManifest(records: [
            DirectTensorTransformRecord(
                sourceName: "weight", destinationName: "weight.packed",
                sourceDataType: .uint8, destinationDataType: .uint4,
                logicalShape: [8], transform: .affineInt4Pack,
                groupSize: 64, quantizationBits: 4, owningStage: .dit),
        ])
        try valid.validate(index: index)

        var invalid = valid
        invalid.records[0].groupSize = 32
        #expect(throws: MappedSafetensorError.self) {
            try invalid.validate(index: index)
        }
    }

    private func makeSingleTensor() -> Data {
        let header = Data(#"{"weight":{"data_offsets":[0,8],"dtype":"U8","shape":[8]}}"#.utf8)
        var length = UInt64(header.count).littleEndian
        var data = Data(bytes: &length, count: 8)
        data.append(header)
        data.append(Data(repeating: 0, count: 8))
        return data
    }
}
