import Testing
import Foundation
@testable import ImarelloCore

@Suite("App cache paths")
struct AppCacheTests {
    @Test("product folder is Imarello and legacy is Aestrix")
    func folderNames() {
        #expect(AppCache.productFolder == "Imarello")
        #expect(AppCache.legacyFolder == "Aestrix")
        #expect(AppCache.directory("models").lastPathComponent == "models")
        #expect(AppCache.directory("models").path.contains("Imarello"))
    }
}
