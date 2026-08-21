import Foundation

actor GalleryStorageService {
    func bytes(for urls: [URL]) -> Int64 {
        urls.reduce(into: 0) { total, url in
            guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
                  let size = attributes[.size] as? NSNumber
            else { return }
            total += size.int64Value
        }
    }
}
