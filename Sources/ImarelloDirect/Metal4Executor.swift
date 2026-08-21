import Dispatch
import Foundation
import Metal
import os

public enum Metal4ExecutorError: Error, Sendable, Equatable, CustomStringConvertible {
    case unavailable(String)
    case tooManyBindings(Int)
    case invalidBindingSlot(Int)
    case duplicateBindingSlot(Int)
    case constantArenaExhausted(required: Int, capacity: Int)
    case feedbackMissing

    public var description: String {
        switch self {
        case .unavailable(let reason): "Metal 4 executor unavailable: \(reason)"
        case .tooManyBindings(let count): "Metal 4 dispatch has \(count) bindings; maximum is 31"
        case .invalidBindingSlot(let slot): "invalid Metal 4 binding slot: \(slot)"
        case .duplicateBindingSlot(let slot): "duplicate Metal 4 binding slot: \(slot)"
        case .constantArenaExhausted(let required, let capacity):
            "constant arena requires \(required) bytes; capacity is \(capacity)"
        case .feedbackMissing: "Metal 4 commit completed without feedback"
        }
    }
}

public struct Metal4DispatchBinding {
    public var slot: Int
    public var buffer: MTLBuffer
    public var byteOffset: Int

    public init(slot: Int, buffer: MTLBuffer, byteOffset: Int = 0) {
        self.slot = slot
        self.buffer = buffer
        self.byteOffset = byteOffset
    }
}

public struct Metal4DispatchTiming: Sendable, Equatable {
    public var gpuStartTime: TimeInterval
    public var gpuEndTime: TimeInterval
    public var gpuDuration: TimeInterval { max(0, gpuEndTime - gpuStartTime) }
}

/// Aligned replacement for legacy `setBytes` calls. The arena is reset only
/// by `Metal4Executor` after commit feedback confirms GPU completion.
@available(macOS 26.0, iOS 26.0, *)
public final class Metal4ConstantArena {
    public let buffer: MTLBuffer
    public let capacity: Int
    public let minimumAlignment: Int
    public private(set) var cursor = 0

    public init(
        device: MTLDevice,
        capacity: Int = 64 * 1_024,
        minimumAlignment: Int = 256
    ) throws {
        guard capacity > 0,
              minimumAlignment > 0,
              minimumAlignment & (minimumAlignment - 1) == 0,
              let buffer = device.makeBuffer(length: capacity, options: [.storageModeShared])
        else { throw Metal4ExecutorError.unavailable("constant arena allocation") }
        self.buffer = buffer
        self.capacity = capacity
        self.minimumAlignment = minimumAlignment
        buffer.label = "metal4.constants"
    }

    public func write<T>(_ value: T, alignment: Int? = nil) throws -> Metal4DispatchBinding {
        let requestedAlignment = max(
            minimumAlignment, alignment ?? MemoryLayout<T>.alignment)
        let offset = try Self.requiredOffset(
            cursor: cursor,
            alignment: requestedAlignment,
            capacity: capacity,
            byteCount: MemoryLayout<T>.size)
        var copy = value
        withUnsafeBytes(of: &copy) { bytes in
            buffer.contents().advanced(by: offset)
                .copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
        cursor = offset + MemoryLayout<T>.size
        return Metal4DispatchBinding(slot: -1, buffer: buffer, byteOffset: offset)
    }

    static func requiredOffset(
        cursor: Int, alignment: Int, capacity: Int, byteCount: Int
    ) throws -> Int {
        guard cursor >= 0, alignment > 0,
              alignment & (alignment - 1) == 0,
              byteCount >= 0
        else { throw Metal4ExecutorError.unavailable("invalid constant allocation") }
        let aligned = (cursor + alignment - 1) / alignment * alignment
        guard aligned <= capacity, byteCount <= capacity - aligned else {
            throw Metal4ExecutorError.constantArenaExhausted(
                required: aligned + byteCount, capacity: capacity)
        }
        return aligned
    }

    fileprivate func resetAfterCompletion() {
        cursor = 0
    }
}

/// Reversible Metal 4 compute path used for isolated qualification. It is not
/// selected by the product until tensor, full-consumer, reliability, and
/// end-to-end gates pass.
#if !targetEnvironment(simulator)
@available(macOS 26.0, iOS 26.0, *)
public final class Metal4Executor {
    public let device: MTLDevice
    public let constantArena: Metal4ConstantArena

    private let queue: MTL4CommandQueue
    private let allocator: MTL4CommandAllocator
    private let argumentTable: MTL4ArgumentTable
    private let residencySet: MTLResidencySet

    public init(device: MTLDevice, constantArenaBytes: Int = 64 * 1_024) throws {
        guard let queue = device.makeMTL4CommandQueue(),
              let allocator = device.makeCommandAllocator()
        else { throw Metal4ExecutorError.unavailable("queue or command allocator") }
        let tableDescriptor = MTL4ArgumentTableDescriptor()
        tableDescriptor.maxBufferBindCount = DirectKernelABI.metal4ArgumentTableBufferLimit
        tableDescriptor.initializeBindings = true
        tableDescriptor.label = "imarello.direct.arguments"
        let argumentTable = try device.makeArgumentTable(descriptor: tableDescriptor)
        let residencyDescriptor = MTLResidencySetDescriptor()
        residencyDescriptor.label = "imarello.direct.dispatch-residency"
        residencyDescriptor.initialCapacity = DirectKernelABI.metal4ArgumentTableBufferLimit
        let residencySet = try device.makeResidencySet(descriptor: residencyDescriptor)

        self.device = device
        self.queue = queue
        self.allocator = allocator
        self.argumentTable = argumentTable
        self.residencySet = residencySet
        self.constantArena = try Metal4ConstantArena(
            device: device, capacity: constantArenaBytes)
    }

    public func dispatchThreadgroups(
        pipeline: MTLComputePipelineState,
        bindings: [Metal4DispatchBinding],
        threadgroupsPerGrid: MTLSize,
        threadsPerThreadgroup: MTLSize,
        label: String
    ) throws -> Metal4DispatchTiming {
        guard bindings.count <= DirectKernelABI.metal4ArgumentTableBufferLimit else {
            throw Metal4ExecutorError.tooManyBindings(bindings.count)
        }
        var slots = Set<Int>()
        for binding in bindings {
            guard (0 ..< DirectKernelABI.metal4ArgumentTableBufferLimit)
                .contains(binding.slot),
                  binding.byteOffset >= 0,
                  binding.byteOffset <= binding.buffer.length
            else { throw Metal4ExecutorError.invalidBindingSlot(binding.slot) }
            guard slots.insert(binding.slot).inserted else {
                throw Metal4ExecutorError.duplicateBindingSlot(binding.slot)
            }
        }

        for slot in 0 ..< DirectKernelABI.metal4ArgumentTableBufferLimit {
            argumentTable.setAddress(0, index: slot)
        }
        for binding in bindings {
            argumentTable.setAddress(
                binding.buffer.gpuAddress + UInt64(binding.byteOffset),
                index: binding.slot)
        }

        let retainedBuffers = uniqueBuffers(bindings.map(\.buffer))
        residencySet.removeAllAllocations()
        for buffer in retainedBuffers { residencySet.addAllocation(buffer) }
        residencySet.commit()

        guard let commandBuffer = device.makeCommandBuffer() else {
            throw Metal4ExecutorError.unavailable("command buffer")
        }
        commandBuffer.label = label
        commandBuffer.beginCommandBuffer(allocator: allocator)
        commandBuffer.useResidencySet(residencySet)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw Metal4ExecutorError.unavailable("compute encoder")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setArgumentTable(argumentTable)
        encoder.dispatchThreadgroups(
            threadgroupsPerGrid: threadgroupsPerGrid,
            threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        commandBuffer.endCommandBuffer()

        let semaphore = DispatchSemaphore(value: 0)
        let feedback = OSAllocatedUnfairLock<Metal4DispatchTiming?>(initialState: nil)
        let options = MTL4CommitOptions()
        options.addFeedbackHandler { value in
            let timing = Metal4DispatchTiming(
                gpuStartTime: value.gpuStartTime,
                gpuEndTime: value.gpuEndTime)
            feedback.withLock { $0 = timing }
            semaphore.signal()
        }
        queue.commit([commandBuffer], options: options)
        semaphore.wait()

        let result = feedback.withLock { $0 }
        withExtendedLifetime(retainedBuffers) {}
        allocator.reset()
        constantArena.resetAfterCompletion()
        residencySet.removeAllAllocations()
        residencySet.commit()
        guard let result else { throw Metal4ExecutorError.feedbackMissing }
        return result
    }

    private func uniqueBuffers(_ buffers: [MTLBuffer]) -> [MTLBuffer] {
        var seen = Set<ObjectIdentifier>()
        return buffers.filter { seen.insert(ObjectIdentifier($0)).inserted }
    }
}
#endif
