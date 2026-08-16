import Testing
import Foundation
@testable import ImarelloCore

@Suite("Text token mode")
struct TextTokenModeTests {
    @Test("raw values are 512 and auto")
    func rawValues() {
        #expect(TextTokenMode.full512.rawValue == "512")
        #expect(TextTokenMode.auto.rawValue == "auto")
        #expect(TextTokenMode(rawValue: "512") == .full512)
        #expect(TextTokenMode(rawValue: "auto") == .auto)
        #expect(TextTokenMode(rawValue: "256") == nil)
    }

    @Test("library request default is pad-512")
    func requestDefaultIsPad512() {
        let t2i = T2IRequest(prompt: "x")
        #expect(t2i.textTokens == .full512)
        let i2i = I2IRequest(prompt: "x", imageURL: URL(fileURLWithPath: "/tmp/ref.png"))
        #expect(i2i.textTokens == .full512)
    }
}
