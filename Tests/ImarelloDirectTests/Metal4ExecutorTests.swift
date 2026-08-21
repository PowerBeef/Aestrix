import Testing
@testable import ImarelloDirect

@Suite("Metal 4 executor contracts")
struct Metal4ExecutorTests {
    @Test("constant upload offsets are aligned and fail closed at capacity")
    func constantOffsets() throws {
        #expect(try Metal4ConstantArena.requiredOffset(
            cursor: 1, alignment: 256, capacity: 1_024, byteCount: 4) == 256)
        #expect(throws: Metal4ExecutorError.constantArenaExhausted(
            required: 1_028, capacity: 1_024)) {
            try Metal4ConstantArena.requiredOffset(
                cursor: 1_023, alignment: 256, capacity: 1_024, byteCount: 4)
        }
    }
}
