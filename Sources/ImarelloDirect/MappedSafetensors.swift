import Darwin
import Foundation
import Metal
import ImarelloPlan

public enum DirectTensorTransform: String, Codable, Sendable, CaseIterable {
    case identity
    case cast
    case transpose
    case affineInt4Pack
}

public struct DirectTensorTransformRecord: Codable, Sendable, Equatable {
    public var sourceName: String
    public var destinationName: String
    public var sourceDataType: SafetensorDataType
    public var destinationDataType: PlanDataType
    public var logicalShape: [Int]
    public var transform: DirectTensorTransform
    public var groupSize: Int?
    public var quantizationBits: Int?
    public var owningStage: PlanStage

    public init(
        sourceName: String,
        destinationName: String,
        sourceDataType: SafetensorDataType,
        destinationDataType: PlanDataType,
        logicalShape: [Int],
        transform: DirectTensorTransform,
        groupSize: Int? = nil,
        quantizationBits: Int? = nil,
        owningStage: PlanStage
    ) {
        self.sourceName = sourceName
        self.destinationName = destinationName
        self.sourceDataType = sourceDataType
        self.destinationDataType = destinationDataType
        self.logicalShape = logicalShape
        self.transform = transform
        self.groupSize = groupSize
        self.quantizationBits = quantizationBits
        self.owningStage = owningStage
    }
}

public struct DirectTensorTransformManifest: Codable, Sendable, Equatable {
    public var schemaVersion: Int
    public var records: [DirectTensorTransformRecord]

    public init(schemaVersion: Int = 1, records: [DirectTensorTransformRecord]) {
        self.schemaVersion = schemaVersion
        self.records = records.sorted { $0.sourceName < $1.sourceName }
    }

    public func validate(index: SafetensorIndex) throws {
        guard schemaVersion == 1 else {
            throw MappedSafetensorError.invalidTransformManifest("schema \(schemaVersion)")
        }
        let recordsByName = Dictionary(grouping: records, by: \.sourceName)
        if let duplicate = recordsByName.first(where: { $0.value.count != 1 })?.key {
            throw MappedSafetensorError.invalidTransformManifest(
                "duplicate source \(duplicate)")
        }
        let tensors = index.tensorsByName
        for record in records {
            guard let descriptor = tensors[record.sourceName] else {
                throw SafetensorError.missingTensor(record.sourceName)
            }
            guard descriptor.dataType == record.sourceDataType,
                  descriptor.shape == record.logicalShape,
                  !record.destinationName.isEmpty
            else {
                throw MappedSafetensorError.invalidTransformManifest(record.sourceName)
            }
            if record.transform == .affineInt4Pack {
                guard record.groupSize == 64, record.quantizationBits == 4 else {
                    throw MappedSafetensorError.invalidTransformManifest(record.sourceName)
                }
            }
        }
    }
}

/// Page-rounded file mapping shared by one or more tensor views.
/// This reference is intentionally non-Sendable; Metal resources and mapping
/// lifetime stay in the runtime actor that created them.
public final class MappedSafetensorSegment {
    public let fileURL: URL
    public let mappedFileRange: Range<Int>
    public let tensorOffsets: [String: Int]
    public let tensorByteCounts: [String: Int]

    private let baseAddress: UnsafeMutableRawPointer
    private let mappedLength: Int

    fileprivate init(
        fileURL: URL,
        descriptors: [SafetensorDescriptor],
        fileSize: Int
    ) throws {
        guard let first = descriptors.map(\.byteRange.lowerBound).min(),
              let last = descriptors.map(\.byteRange.upperBound).max(),
              first >= 0, last <= fileSize, first < last
        else { throw MappedSafetensorError.emptySelection }

        let page = Int(getpagesize())
        let mapStart = first / page * page
        let requestedEnd = ((last + page - 1) / page) * page
        let mapEnd = max(mapStart + page, requestedEnd)
        let length = mapEnd - mapStart
        let descriptor = open(fileURL.path, O_RDONLY | O_CLOEXEC)
        guard descriptor >= 0 else {
            throw MappedSafetensorError.openFailed(String(cString: strerror(errno)))
        }
        defer { close(descriptor) }
        let address = mmap(nil, length, PROT_READ, MAP_PRIVATE, descriptor, off_t(mapStart))
        guard address != MAP_FAILED, let address else {
            throw MappedSafetensorError.mapFailed(String(cString: strerror(errno)))
        }

        self.fileURL = fileURL
        self.mappedFileRange = mapStart ..< mapEnd
        self.baseAddress = address
        self.mappedLength = length
        self.tensorOffsets = Dictionary(uniqueKeysWithValues: descriptors.map {
            ($0.name, $0.byteRange.lowerBound - mapStart)
        })
        self.tensorByteCounts = Dictionary(uniqueKeysWithValues: descriptors.map {
            ($0.name, $0.byteRange.count)
        })
    }

    deinit {
        munmap(baseAddress, mappedLength)
    }

    public func withUnsafeTensorBytes<T>(
        named name: String,
        _ body: (UnsafeRawBufferPointer) throws -> T
    ) rethrows -> T? {
        guard let offset = tensorOffsets[name], let count = tensorByteCounts[name] else {
            return nil
        }
        return try body(UnsafeRawBufferPointer(
            start: UnsafeRawPointer(baseAddress.advanced(by: offset)), count: count))
    }

    public func makeMetalBuffer(device: MTLDevice) throws -> MappedMetalBuffer {
        guard let buffer = device.makeBuffer(
            bytesNoCopy: baseAddress,
            length: mappedLength,
            options: [.storageModeShared],
            deallocator: nil)
        else { throw MappedSafetensorError.metalBufferFailed }
        buffer.label = "mmap:\(fileURL.lastPathComponent)"
        return MappedMetalBuffer(buffer: buffer, segment: self)
    }
}

/// Releases the Metal view before releasing the mapping that backs it.
/// This object remains actor-confined and deliberately non-Sendable.
public final class MappedMetalBuffer {
    private var storage: MTLBuffer?
    private let segment: MappedSafetensorSegment

    fileprivate init(buffer: MTLBuffer, segment: MappedSafetensorSegment) {
        self.storage = buffer
        self.segment = segment
    }

    public var buffer: MTLBuffer {
        precondition(storage != nil, "mapped Metal buffer accessed after release")
        return storage!
    }

    public var tensorOffsets: [String: Int] { segment.tensorOffsets }

    deinit {
        storage = nil
    }
}

public struct MappedSafetensorFile {
    public let url: URL
    public let index: SafetensorIndex

    public init(
        url: URL,
        allowedTensorNames: Set<String>? = nil,
        requiredTensorNames: Set<String> = []
    ) throws {
        self.url = url
        self.index = try SafetensorIndex.parse(
            url: url,
            allowedTensorNames: allowedTensorNames,
            requiredTensorNames: requiredTensorNames)
    }

    public func map(tensorNames: Set<String>) throws -> MappedSafetensorSegment {
        guard !tensorNames.isEmpty else { throw MappedSafetensorError.emptySelection }
        let byName = index.tensorsByName
        let descriptors = try tensorNames.sorted().map { name -> SafetensorDescriptor in
            guard let descriptor = byName[name] else {
                throw SafetensorError.missingTensor(name)
            }
            return descriptor
        }
        return try MappedSafetensorSegment(
            fileURL: url, descriptors: descriptors, fileSize: index.fileSize)
    }
}

public enum MappedSafetensorError: Error, Sendable, Equatable, CustomStringConvertible {
    case emptySelection
    case openFailed(String)
    case mapFailed(String)
    case metalBufferFailed
    case invalidTransformManifest(String)

    public var description: String {
        switch self {
        case .emptySelection: "no tensors selected for mapping"
        case .openFailed(let reason): "failed to open safetensors file: \(reason)"
        case .mapFailed(let reason): "failed to mmap safetensors file: \(reason)"
        case .metalBufferFailed: "Metal rejected the page-rounded no-copy mapping"
        case .invalidTransformManifest(let reason):
            "invalid Direct transform manifest: \(reason)"
        }
    }
}
