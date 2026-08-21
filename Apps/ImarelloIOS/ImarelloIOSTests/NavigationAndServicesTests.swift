import Foundation
import Photos
import Testing
import UIKit
@testable import Imarello
import ImarelloCore

@MainActor
@Suite("Navigation and services")
struct NavigationAndServicesTests {
    @Test("tabs retain independent navigation histories")
    func independentHistory() {
        let navigation = AppNavigation()
        navigation.openImage(id: "create", from: .create)
        navigation.openImage(id: "edit", from: .edit)
        navigation.openImage(id: "gallery", from: .gallery)
        navigation.openImage(id: "settings", from: .settings)

        #expect(navigation.createPath == [.image(id: "create")])
        #expect(navigation.editPath == [.image(id: "edit")])
        #expect(navigation.galleryPath == [.image(id: "gallery")])
        #expect(navigation.settingsPath == [.image(id: "settings")])
        navigation.showCreateRoot()
        #expect(navigation.createPath.isEmpty)
        #expect(navigation.editPath == [.image(id: "edit")])
        #expect(navigation.galleryPath == [.image(id: "gallery")])
        #expect(navigation.settingsPath == [.image(id: "settings")])
        #expect(AppTab.allCases == [.create, .edit, .gallery, .settings])
    }

    @Test("activity routes to the originating workspace")
    func activityDestination() {
        let source = PrintRecord(
            id: "source", fileName: "source.png", prompt: "", seed: 1,
            side: 512, mode: "t2i", createdAt: .now
        )
        let create = GenerationOperation.create(
            GenerationRequestSnapshot(prompt: "create", side: 1024, seed: 42, source: nil)
        )
        let edit = GenerationOperation.edit(
            GenerationRequestSnapshot(prompt: "edit", side: 512, seed: 7, source: source)
        )
        #expect(create.destination == .create)
        #expect(edit.destination == .edit)
    }

    @Test("harness monitor owns exactly one active poll task")
    func harnessLifecycle() {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("HarnessMonitorTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let store = PrintStore(
            durableRoot: root.appendingPathComponent("durable"),
            outputsDirectory: root.appendingPathComponent("outputs"),
            legacyIndexURL: root.appendingPathComponent("legacy.json"),
            fileIO: SystemPrintStoreFileIO()
        )
        let engine = MonitorGenerationService(root: root)
        let session = StudioSession(engine: engine, store: store)
        var polls = 0
        let monitor = HarnessMonitor(session: session) { _ in polls += 1 }

        monitor.setActive(true)
        monitor.setActive(true)
        #expect(monitor.isMonitoring)
        #expect(polls == 1)
        monitor.setActive(false)
        #expect(!monitor.isMonitoring)
    }

    @Test("image loader caches and honors pre-cancellation")
    func imageLoader() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ImageLoaderTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let url = root.appendingPathComponent("fixture.png")
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: 64, height: 64))
        try #require(renderer.image { _ in UIColor.red.setFill(); UIRectFill(CGRect(x: 0, y: 0, width: 64, height: 64)) }.pngData()).write(to: url)

        let loader = PrintImageLoader()
        let first = try await loader.image(at: url, recordID: "fixture", maxPixel: 64)
        let second = try await loader.image(at: url, recordID: "fixture", maxPixel: 64)
        #expect(first === second)

        let cancelled = Task {
            try await loader.image(at: url, recordID: "cancelled", maxPixel: 64)
        }
        cancelled.cancel()
        await #expect(throws: CancellationError.self) {
            _ = try await cancelled.value
        }
    }

    @Test("Gallery storage is calculated off the main actor")
    func galleryStorage() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("GalleryStorageTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let first = root.appendingPathComponent("first.png")
        let second = root.appendingPathComponent("second.png")
        try Data(repeating: 1, count: 128).write(to: first)
        try Data(repeating: 2, count: 256).write(to: second)

        let bytes = await GalleryStorageService().bytes(for: [first, second])
        #expect(bytes == 384)
    }

    @Test("Photos status maps to a useful Settings presentation")
    func photosPresentation() {
        #expect(PhotoAccessPresentation(status: .authorized).title == "Allowed")
        #expect(!PhotoAccessPresentation(status: .authorized).canOpenSettings)
        #expect(PhotoAccessPresentation(status: .denied).canOpenSettings)
        #expect(PhotoAccessPresentation(status: .restricted).canOpenSettings)
    }

    @Test("Edit sources feature the newest image without duplicating it")
    func editSourceCollection() {
        let oldest = PrintRecord(
            id: "oldest", fileName: "oldest.png", prompt: "", seed: 1,
            side: 512, mode: "t2i", createdAt: .now
        )
        let middle = PrintRecord(
            id: "middle", fileName: "middle.png", prompt: "", seed: 2,
            side: 1024, mode: "i2i", createdAt: .now
        )
        let newest = PrintRecord(
            id: "newest", fileName: "newest.png", prompt: "", seed: 3,
            side: 1024, mode: "i2i", createdAt: .now
        )

        let empty = EditSourceCollection(records: [])
        #expect(empty.featured == nil)
        #expect(empty.remaining.isEmpty)

        let single = EditSourceCollection(records: [newest])
        #expect(single.featured == newest)
        #expect(single.remaining.isEmpty)

        let many = EditSourceCollection(records: [newest, middle, oldest])
        #expect(many.featured == newest)
        #expect(many.remaining == [middle, oldest])
        #expect(!many.remaining.contains(newest))
    }
}

@MainActor
private final class MonitorGenerationService: GenerationServing {
    var gate: GenerationEngine.RunGate = .ready
    var allowsGenerationInCurrentProcess = false
    let expectedModelsDirectory: URL

    init(root: URL) { expectedModelsDirectory = root }
    func refreshGate() {}
    func generate(
        prompt: String, side: Int, seed: UInt64, steps: Int,
        textTokens: TextTokenMode,
        onProgress: @escaping @MainActor (PipelineProgress) -> Void
    ) async throws -> URL { throw CancellationError() }
    func edit(
        source: URL, prompt: String, side: Int, seed: UInt64,
        strength: Float, steps: Int, textTokens: TextTokenMode,
        onProgress: @escaping @MainActor (PipelineProgress) -> Void
    ) async throws -> URL { throw CancellationError() }
}
