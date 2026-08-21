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
        var blob = Data(repeating: 0, count: MetallibVerification.fullPackByteThreshold)
        blob.append(contentsOf: "xxsteel_attentionxxaffine_qmmxxrms_normxx".utf8)
        let v = MetallibVerification.verify(data: blob, path: "full")
        #expect(!v.isStub)
        #expect(v.productReady)
        #expect(v.steelSymbolsMissing.isEmpty)
        #expect(!v.naxPackaged)
    }

    @Test("Xcode Cmlx JIT lib is rejected without packaged affine_qmm")
    func jitCmlxWithoutAffineQmmIsRejected() {
        var blob = Data(repeating: 0, count: MetallibVerification.stubByteThreshold + 16)
        blob.append(contentsOf: "xxsteel_attentionxxrms_normxx".utf8)
        let v = MetallibVerification.verify(data: blob, path: "cmlx-ios")
        #expect(!v.productReady)
        #expect(v.steelSymbolsMissing == ["affine_qmm"])
    }

    @Test("pickBest ignores a system default.metallib stub")
    func pickBestSkipsSystemStub() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("imarello-metallib-pick-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let stub = dir.appendingPathComponent("default.metallib")
        try Data(repeating: 0xAB, count: 156_964).write(to: stub)

        let cmlx = dir.appendingPathComponent("mlx-cmlx.metallib")
        var blob = Data(repeating: 0, count: MetallibVerification.fullPackByteThreshold)
        blob.append(contentsOf: "xxsteel_attentionxxaffine_qmmxxrms_normxx".utf8)
        try blob.write(to: cmlx)

        let picked = MetallibVerification.pickBest(among: [stub, cmlx])
        #expect(picked?.resolvingSymlinksInPath().path == cmlx.resolvingSymlinksInPath().path)

        let onlyStub = MetallibVerification.pickBest(among: [stub])
        #expect(onlyStub == nil)
    }

    @Test("on-disk walk finds nested mlx-swift_Cmlx.bundle")
    func onDiskFindsNestedCmlxBundle() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("imarello-metallib-app-\(UUID().uuidString)", isDirectory: true)
        let bundle = dir.appendingPathComponent("mlx-swift_Cmlx.bundle", isDirectory: true)
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        var blob = Data(repeating: 0, count: MetallibVerification.fullPackByteThreshold)
        blob.append(contentsOf: "xxsteel_attentionxxaffine_qmmxxrms_normxx".utf8)
        let metallib = bundle.appendingPathComponent("default.metallib")
        try blob.write(to: metallib)

        let found = MetallibVerification.onDiskMetallibs(in: dir)
        let picked = MetallibVerification.pickBest(among: found)
        #expect(found.contains { $0.resolvingSymlinksInPath().path == metallib.resolvingSymlinksInPath().path })
        #expect(picked?.resolvingSymlinksInPath().path == metallib.resolvingSymlinksInPath().path)
    }

    @Test("on-disk walk of the iphoneos Imarello.app finds Cmlx")
    func onDiskFindsBuiltIOSApp() throws {
        let app = URL(fileURLWithPath: "/tmp/ImarelloIOS-iphoneos-dd/Build/Products/Debug-iphoneos/Imarello.app")
        guard FileManager.default.fileExists(atPath: app.path) else { return }
        let found = MetallibVerification.onDiskMetallibs(in: app)
        let picked = MetallibVerification.pickBest(among: found)
        #expect(picked != nil)
        let check = MetallibVerification.verify(url: picked!)
        #expect(check.productReady)
        #expect(picked!.path.contains("mlx-swift_Cmlx.bundle"))
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
