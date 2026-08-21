import Foundation
import Observation
import Photos

enum PhotoExportError: LocalizedError {
    case busy
    case denied

    var errorDescription: String? {
        switch self {
        case .busy: return "Another image is already being saved."
        case .denied: return "Photos access was denied. Enable it in Settings to save."
        }
    }
}

@MainActor
@Observable
final class PhotoExportService {
    private(set) var isSaving = false

    func save(_ url: URL) async throws {
        guard !isSaving else { throw PhotoExportError.busy }
        isSaving = true
        defer { isSaving = false }

        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else {
            throw PhotoExportError.denied
        }
        try await PHPhotoLibrary.shared().performChanges {
            PHAssetChangeRequest.creationRequestForAssetFromImage(atFileURL: url)
        }
    }
}
