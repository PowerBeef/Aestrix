import Foundation
import ImageIO
import SwiftUI
import UIKit

actor PrintImageLoader {
    private let cache = NSCache<NSString, UIImage>()

    init() {
        cache.countLimit = 32
        cache.totalCostLimit = 96 * 1_024 * 1_024
    }

    func image(at url: URL, recordID: String, maxPixel: Int? = nil) throws -> UIImage? {
        try Task.checkCancellation()
        let bucket = maxPixel.map(String.init) ?? "full"
        let key = "\(recordID)#\(bucket)" as NSString
        if let image = cache.object(forKey: key) { return image }

        let image: UIImage?
        if let maxPixel {
            let options: [CFString: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: maxPixel * 2,
                kCGImageSourceShouldCacheImmediately: true,
            ]
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let cgImage = CGImageSourceCreateThumbnailAtIndex(
                    source, 0, options as CFDictionary
                  )
            else { return nil }
            image = UIImage(cgImage: cgImage)
        } else {
            let options: [CFString: Any] = [
                kCGImageSourceShouldCache: false,
                kCGImageSourceShouldCacheImmediately: true,
            ]
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let cgImage = CGImageSourceCreateImageAtIndex(
                    source, 0, options as CFDictionary
                  )
            else { return nil }
            image = UIImage(cgImage: cgImage)
        }

        try Task.checkCancellation()
        if let image {
            let cost = Int(image.size.width * image.size.height * image.scale * image.scale * 4)
            cache.setObject(image, forKey: key, cost: cost)
        }
        return image
    }

    func remove(recordID: String) {
        for bucket in ["240", "512", "1024", "full"] {
            cache.removeObject(forKey: "\(recordID)#\(bucket)" as NSString)
        }
    }

    func removeAll() {
        cache.removeAllObjects()
    }
}

private struct PrintImageLoaderKey: EnvironmentKey {
    static let defaultValue = PrintImageLoader()
}

extension EnvironmentValues {
    var printImageLoader: PrintImageLoader {
        get { self[PrintImageLoaderKey.self] }
        set { self[PrintImageLoaderKey.self] = newValue }
    }
}
