import Foundation
import CoreFoundation

public enum SafetensorDataType: String, Codable, Sendable, CaseIterable {
    case bool = "BOOL"
    case uint8 = "U8"
    case int8 = "I8"
    case uint16 = "U16"
    case int16 = "I16"
    case float16 = "F16"
    case bfloat16 = "BF16"
    case uint32 = "U32"
    case int32 = "I32"
    case float32 = "F32"
    case uint64 = "U64"
    case int64 = "I64"
    case float64 = "F64"

    public var byteWidth: Int {
        switch self {
        case .bool, .uint8, .int8: 1
        case .uint16, .int16, .float16, .bfloat16: 2
        case .uint32, .int32, .float32: 4
        case .uint64, .int64, .float64: 8
        }
    }
}

public struct SafetensorDescriptor: Codable, Sendable, Equatable {
    public var name: String
    public var dataType: SafetensorDataType
    public var shape: [Int]
    /// Absolute byte range in the containing file.
    public var byteRange: Range<Int>
}

public struct SafetensorIndex: Sendable, Equatable {
    public static let maximumHeaderBytes = 128 * 1_024 * 1_024

    public var fileSize: Int
    public var payloadOffset: Int
    public var tensors: [SafetensorDescriptor]

    public var tensorsByName: [String: SafetensorDescriptor] {
        Dictionary(uniqueKeysWithValues: tensors.map { ($0.name, $0) })
    }

    public static func parse(
        url: URL,
        allowedTensorNames: Set<String>? = nil,
        requiredTensorNames: Set<String> = []
    ) throws -> SafetensorIndex {
        let data: Data
        do {
            data = try Data(contentsOf: url, options: [.mappedIfSafe])
        } catch {
            throw SafetensorError.unreadable(error.localizedDescription)
        }
        return try parse(
            data: data,
            allowedTensorNames: allowedTensorNames,
            requiredTensorNames: requiredTensorNames)
    }

    public static func parse(
        data: Data,
        allowedTensorNames: Set<String>? = nil,
        requiredTensorNames: Set<String> = []
    ) throws -> SafetensorIndex {
        guard data.count >= 8 else { throw SafetensorError.truncatedPrefix }
        let headerLength = data.prefix(8).enumerated().reduce(UInt64(0)) { partial, item in
            partial | UInt64(item.element) << UInt64(item.offset * 8)
        }
        guard headerLength > 0,
              headerLength <= UInt64(maximumHeaderBytes),
              headerLength <= UInt64(Int.max - 8)
        else { throw SafetensorError.invalidHeaderLength(headerLength) }
        let payloadOffset = 8 + Int(headerLength)
        guard payloadOffset <= data.count else {
            throw SafetensorError.truncatedHeader(
                expectedEnd: payloadOffset, fileSize: data.count)
        }

        let header = data.subdata(in: 8 ..< payloadOffset)
        if let duplicate = try duplicateTopLevelKey(in: header) {
            throw SafetensorError.duplicateTensor(duplicate)
        }
        let object: [String: Any]
        do {
            guard let decoded = try JSONSerialization.jsonObject(with: header) as? [String: Any]
            else { throw SafetensorError.invalidHeaderJSON }
            object = decoded
        } catch let error as SafetensorError {
            throw error
        } catch {
            throw SafetensorError.invalidHeaderJSON
        }

        let payloadBytes = data.count - payloadOffset
        var descriptors = [SafetensorDescriptor]()
        descriptors.reserveCapacity(object.count)
        for (name, rawDescriptor) in object where name != "__metadata__" {
            if let allowedTensorNames, !allowedTensorNames.contains(name) {
                throw SafetensorError.unexpectedTensor(name)
            }
            guard let dictionary = rawDescriptor as? [String: Any],
                  let rawDType = dictionary["dtype"] as? String,
                  let dataType = SafetensorDataType(rawValue: rawDType),
                  let rawShape = dictionary["shape"] as? [Any],
                  let rawOffsets = dictionary["data_offsets"] as? [Any],
                  rawOffsets.count == 2
            else { throw SafetensorError.invalidDescriptor(name) }

            let shape = try rawShape.map { value -> Int in
                guard let number = value as? NSNumber,
                      CFGetTypeID(number) != CFBooleanGetTypeID(),
                      number.int64Value >= 0,
                      UInt64(number.int64Value) <= UInt64(Int.max)
                else { throw SafetensorError.invalidShape(name) }
                return Int(number.int64Value)
            }
            let offsets = try rawOffsets.map { value -> Int in
                guard let number = value as? NSNumber,
                      CFGetTypeID(number) != CFBooleanGetTypeID(),
                      number.int64Value >= 0,
                      UInt64(number.int64Value) <= UInt64(Int.max)
                else { throw SafetensorError.invalidOffsets(name) }
                return Int(number.int64Value)
            }
            guard offsets[0] <= offsets[1], offsets[1] <= payloadBytes else {
                throw SafetensorError.outOfBounds(name)
            }

            let elementCount = try checkedElementCount(shape, name: name)
            let expectedBytes = try checkedMultiply(
                elementCount, dataType.byteWidth, name: name)
            guard offsets[1] - offsets[0] == expectedBytes else {
                throw SafetensorError.byteCountMismatch(
                    name: name,
                    expected: expectedBytes,
                    actual: offsets[1] - offsets[0])
            }
            descriptors.append(
                SafetensorDescriptor(
                    name: name,
                    dataType: dataType,
                    shape: shape,
                    byteRange: payloadOffset + offsets[0] ..< payloadOffset + offsets[1]))
        }

        let found = Set(descriptors.map(\.name))
        if let missing = requiredTensorNames.subtracting(found).sorted().first {
            throw SafetensorError.missingTensor(missing)
        }
        descriptors.sort {
            if $0.byteRange.lowerBound != $1.byteRange.lowerBound {
                return $0.byteRange.lowerBound < $1.byteRange.lowerBound
            }
            return $0.name < $1.name
        }
        for pair in zip(descriptors, descriptors.dropFirst())
        where pair.1.byteRange.lowerBound < pair.0.byteRange.upperBound {
            throw SafetensorError.overlappingTensors(pair.0.name, pair.1.name)
        }

        return SafetensorIndex(
            fileSize: data.count,
            payloadOffset: payloadOffset,
            tensors: descriptors)
    }

    private static func checkedElementCount(_ shape: [Int], name: String) throws -> Int {
        try shape.reduce(1) { partial, dimension in
            try checkedMultiply(partial, dimension, name: name)
        }
    }

    private static func checkedMultiply(_ lhs: Int, _ rhs: Int, name: String) throws -> Int {
        let result = lhs.multipliedReportingOverflow(by: rhs)
        guard !result.overflow else { throw SafetensorError.shapeOverflow(name) }
        return result.partialValue
    }

    /// JSONSerialization accepts duplicate keys. Safetensors tensor names are
    /// top-level keys, so detect duplicates before decoding the object.
    private static func duplicateTopLevelKey(in data: Data) throws -> String? {
        let bytes = Array(data)
        var index = 0
        var depth = 0
        var expectingKey = false
        var keys = Set<String>()
        while index < bytes.count {
            switch bytes[index] {
            case 0x7B: // {
                depth += 1
                if depth == 1 { expectingKey = true }
                index += 1
            case 0x7D: // }
                depth -= 1
                guard depth >= 0 else { throw SafetensorError.invalidHeaderJSON }
                index += 1
            case 0x2C: // ,
                if depth == 1 { expectingKey = true }
                index += 1
            case 0x22: // "
                let start = index
                index += 1
                var escaped = false
                while index < bytes.count {
                    let byte = bytes[index]
                    if escaped {
                        escaped = false
                    } else if byte == 0x5C {
                        escaped = true
                    } else if byte == 0x22 {
                        break
                    }
                    index += 1
                }
                guard index < bytes.count else { throw SafetensorError.invalidHeaderJSON }
                if depth == 1, expectingKey {
                    let encoded = Data(bytes[start ... index])
                    guard let key = try? JSONDecoder().decode(String.self, from: encoded) else {
                        throw SafetensorError.invalidHeaderJSON
                    }
                    guard keys.insert(key).inserted else { return key }
                    expectingKey = false
                }
                index += 1
            default:
                index += 1
            }
        }
        guard depth == 0 else { throw SafetensorError.invalidHeaderJSON }
        return nil
    }
}

public enum SafetensorError: Error, Sendable, Equatable, CustomStringConvertible {
    case unreadable(String)
    case truncatedPrefix
    case invalidHeaderLength(UInt64)
    case truncatedHeader(expectedEnd: Int, fileSize: Int)
    case invalidHeaderJSON
    case duplicateTensor(String)
    case unexpectedTensor(String)
    case missingTensor(String)
    case invalidDescriptor(String)
    case invalidShape(String)
    case shapeOverflow(String)
    case invalidOffsets(String)
    case outOfBounds(String)
    case byteCountMismatch(name: String, expected: Int, actual: Int)
    case overlappingTensors(String, String)

    public var description: String {
        switch self {
        case .unreadable(let detail): "unreadable safetensors file: \(detail)"
        case .truncatedPrefix: "safetensors file is shorter than its eight-byte prefix"
        case .invalidHeaderLength(let length): "invalid safetensors header length: \(length)"
        case .truncatedHeader(let expected, let actual):
            "truncated safetensors header: expected \(expected) bytes, found \(actual)"
        case .invalidHeaderJSON: "invalid safetensors header JSON"
        case .duplicateTensor(let name): "duplicate tensor: \(name)"
        case .unexpectedTensor(let name): "unexpected tensor: \(name)"
        case .missingTensor(let name): "missing tensor: \(name)"
        case .invalidDescriptor(let name): "invalid tensor descriptor: \(name)"
        case .invalidShape(let name): "invalid tensor shape: \(name)"
        case .shapeOverflow(let name): "tensor shape overflows Int: \(name)"
        case .invalidOffsets(let name): "invalid tensor offsets: \(name)"
        case .outOfBounds(let name): "tensor data is out of bounds: \(name)"
        case .byteCountMismatch(let name, let expected, let actual):
            "tensor \(name) byte count mismatch: expected \(expected), found \(actual)"
        case .overlappingTensors(let first, let second):
            "tensor data overlaps: \(first), \(second)"
        }
    }
}
