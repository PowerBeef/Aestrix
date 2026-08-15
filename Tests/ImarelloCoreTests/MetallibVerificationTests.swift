import Testing
import Foundation
@testable import ImarelloCore

@Suite("Metallib verification")
struct MetallibVerificationTests {
    @Test("tiny payload is a stub and not product-ready")
    func stubIsNotReady() {
        let data = Data("noop".utf8)
        let v = MetallibVerification.verify(data: data, path: "stub")
        #expect(v.isStub)
        #expect(!v.productReady)
        #expect(v.steelSymbolsMissing == MetallibVerification.requiredSteelSymbols)
    }

    @Test("required Steel names make a large blob product-ready")
    func steelSymbolsReady() {
        var blob = Data(repeating: 0, count: MetallibVerification.stubByteThreshold + 16)
        blob.append(contentsOf: "xxsteel_attentionxxaffine_qmmxxrms_normxx".utf8)
        let v = MetallibVerification.verify(data: blob, path: "full")
        #expect(!v.isStub)
        #expect(v.productReady)
        #expect(v.steelSymbolsMissing.isEmpty)
        #expect(!v.naxPackaged)
    }

    @Test("bundle search does not trap")
    func resolveFromBundlesDoesNotTrap() {
        _ = MetallibVerification.resolveFromBundles()
    }

    @Test("missing metallib path is a stub")
    func missingFile() {
        let url = URL(fileURLWithPath: "/tmp/imarello-no-such-mlx.metallib")
        let v = MetallibVerification.verify(url: url)
        #expect(!v.productReady)
        #expect(v.isStub)
        #expect(v.byteCount == 0)
    }
}
