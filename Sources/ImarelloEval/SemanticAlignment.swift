import Foundation
import CoreGraphics
import Vision
import CoreML
import ImarelloCore

/// Prompt–image semantic alignment (P1).
///
/// - **Core ML CLIP** when models exist under the Imarello CLIP cache directory
///   (`image_encoder.mlmodelc` + `text_encoder.mlmodelc`, 512-d L2-normalized embeddings).
/// - **Vision classify proxy** otherwise (always available on Apple platforms):
///   ImageNet-style labels from `VNClassifyImageRequest` overlapped with prompt tokens.
///
/// Neither replaces multimodal vision review; both give automated prompt-adherence signal.
public enum SemanticAlignment {
    public struct Metrics: Sendable, Codable, Equatable {
        /// 0…100 (higher = better prompt–image agreement).
        public var score: Float
        /// `coreml_clip` | `vision_proxy` | `unavailable`
        public var backend: String
        public var available: Bool
        /// Cosine similarity in [−1, 1] when CLIP embeddings used; nil for proxy.
        public var cosine: Float?
        /// Top Vision labels (proxy backend) or empty.
        public var topLabels: [String]
        public var notes: [String]
    }

    /// Default Core ML model directory (Imarello, else leftover Aestrix).
    public static var defaultModelDirectory: URL {
        AppCache.resolvedItem(under: "models", item: "clip-coreml")
    }

    /// True if both Core ML CLIP encoders are present.
    public static func coreMLCLIPAvailable(modelDirectory: URL = defaultModelDirectory) -> Bool {
        let img = modelDirectory.appendingPathComponent("image_encoder.mlmodelc")
        let txt = modelDirectory.appendingPathComponent("text_encoder.mlmodelc")
        var isDir: ObjCBool = false
        let fm = FileManager.default
        return fm.fileExists(atPath: img.path, isDirectory: &isDir) && isDir.boolValue
            && fm.fileExists(atPath: txt.path, isDirectory: &isDir) && isDir.boolValue
    }

    /// Score prompt–image alignment. Prefer Core ML CLIP; fall back to Vision proxy.
    public static func analyze(
        pixels: PixelBuffer,
        prompt: String?,
        cgImage: CGImage?,
        modelDirectory: URL = defaultModelDirectory,
        preferCoreML: Bool = true
    ) -> Metrics {
        let trimmed = prompt?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !trimmed.isEmpty else {
            return Metrics(
                score: 50,
                backend: "unavailable",
                available: false,
                cosine: nil,
                topLabels: [],
                notes: ["No prompt; semantic alignment skipped."]
            )
        }

        if preferCoreML, coreMLCLIPAvailable(modelDirectory: modelDirectory), let cg = cgImage {
            if let clip = try? coreMLCLIPScore(cgImage: cg, prompt: trimmed, modelDirectory: modelDirectory) {
                return clip
            }
        }

        if let cg = cgImage {
            return visionProxyScore(cgImage: cg, prompt: trimmed)
        }

        // No CGImage (rare) — keyword-only weak signal from technical path not available here.
        return Metrics(
            score: 50,
            backend: "unavailable",
            available: false,
            cosine: nil,
            topLabels: [],
            notes: ["Semantic alignment unavailable (no CGImage / CLIP models)."]
        )
    }

    // MARK: - Core ML CLIP

    private static func coreMLCLIPScore(
        cgImage: CGImage,
        prompt: String,
        modelDirectory: URL
    ) throws -> Metrics {
        let imgURL = modelDirectory.appendingPathComponent("image_encoder.mlmodelc")
        let txtURL = modelDirectory.appendingPathComponent("text_encoder.mlmodelc")
        let imageModel = try MLModel(contentsOf: imgURL)
        let textModel = try MLModel(contentsOf: txtURL)

        // Expect 224×224 RGB float32 NCHW or NHWC — try common CLIP layouts.
        let side = 224
        let pixelBuffer = try makeCVPixelBuffer(cgImage: cgImage, side: side)

        let imageInputName = imageModel.modelDescription.inputDescriptionsByName.keys.first
            ?? "image"
        let imageProvider = try MLDictionaryFeatureProvider(dictionary: [
            imageInputName: MLFeatureValue(pixelBuffer: pixelBuffer)
        ])
        let imageOut = try imageModel.prediction(from: imageProvider)
        let imageEmb = try firstMultiArray(from: imageOut)

        // Text: many CLIP Core ML exports take a string or int32 token ids.
        let textInputName = textModel.modelDescription.inputDescriptionsByName.keys.first
            ?? "text"
        let textFeat: MLFeatureValue
        if textModel.modelDescription.inputDescriptionsByName[textInputName]?.type == .string {
            textFeat = MLFeatureValue(string: prompt)
        } else {
            // Fallback: pack a simple bag-of-length feature if model expects multiarray (not ideal).
            throw ImarelloError.imageLoadFailed(
                path: txtURL.path,
                reason: "text_encoder expects non-string input; use vision_proxy or export CLIP with string input"
            )
        }
        let textProvider = try MLDictionaryFeatureProvider(dictionary: [
            textInputName: textFeat
        ])
        let textOut = try textModel.prediction(from: textProvider)
        let textEmb = try firstMultiArray(from: textOut)

        let cos = cosineSimilarity(imageEmb, textEmb)
        // Map cosine ~[0, 0.4] typical CLIP range to 0…100 (clip-score style ×100, clamped).
        let score = max(0, min(100, cos * 100))
        return Metrics(
            score: score,
            backend: "coreml_clip",
            available: true,
            cosine: cos,
            topLabels: [],
            notes: [
                String(format: "Core ML CLIP cosine=%.4f → score=%.1f (models in %@)", cos, score, modelDirectory.path)
            ]
        )
    }

    private static func firstMultiArray(from provider: MLFeatureProvider) throws -> MLMultiArray {
        for name in provider.featureNames {
            if let v = provider.featureValue(for: name)?.multiArrayValue {
                return v
            }
        }
        throw ImarelloError.imageLoadFailed(path: "clip", reason: "no multiarray embedding in model output")
    }

    private static func cosineSimilarity(_ a: MLMultiArray, _ b: MLMultiArray) -> Float {
        let n = min(a.count, b.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< n {
            let x = floatAt(a, i)
            let y = floatAt(b, i)
            dot += x * y
            na += x * x
            nb += y * y
        }
        let d = sqrt(na) * sqrt(nb)
        return d > 1e-8 ? dot / d : 0
    }

    private static func floatAt(_ a: MLMultiArray, _ i: Int) -> Float {
        switch a.dataType {
        case .float32:
            return a[i].floatValue
        case .double:
            return Float(a[i].doubleValue)
        case .float16:
            return a[i].floatValue
        default:
            return a[i].floatValue
        }
    }

    private static func makeCVPixelBuffer(cgImage: CGImage, side: Int) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, side, side,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pb
        )
        guard status == kCVReturnSuccess, let buffer = pb else {
            throw ImarelloError.imageLoadFailed(path: "clip", reason: "CVPixelBufferCreate failed")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let cs = CGColorSpaceCreateDeviceRGB()
        let info = CGBitmapInfo.byteOrder32Little.union(
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)
        )
        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: side,
            height: side,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: cs,
            bitmapInfo: info.rawValue
        ) else {
            throw ImarelloError.imageLoadFailed(path: "clip", reason: "CGContext for CLIP resize failed")
        }
        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: side, height: side))
        return buffer
    }

    // MARK: - Vision classify proxy (always available)

    private static func visionProxyScore(cgImage: CGImage, prompt: String) -> Metrics {
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        let request = VNClassifyImageRequest()
        do {
            try handler.perform([request])
        } catch {
            return Metrics(
                score: 50,
                backend: "unavailable",
                available: false,
                cosine: nil,
                topLabels: [],
                notes: ["VNClassifyImageRequest failed: \(error.localizedDescription)"]
            )
        }

        let observations = (request.results ?? [])
            .prefix(15)
            .map { ($0.identifier.lowercased(), $0.confidence) }

        let labels = observations.map(\.0)
        let promptTokens = tokenize(prompt)
        // Match prompt tokens against label identifiers (split on comma / space).
        var hitWeight: Float = 0
        var totalConf: Float = 0
        for (id, conf) in observations {
            totalConf += conf
            let parts = id.split { !$0.isLetter }.map(String.init)
            for p in parts where p.count >= 3 {
                if promptTokens.contains(p) {
                    hitWeight += conf
                }
            }
            // Multi-word: "coffee mug" style
            for t in promptTokens where t.count >= 4 && id.contains(t) {
                hitWeight += conf * 0.5
            }
        }

        // Color boost: if prompt has color word matching is handled separately; add small
        // base from any hit.
        let raw = totalConf > 1e-6 ? hitWeight / max(0.15, totalConf) : 0
        let score = max(0, min(100, 25 + raw * 75))

        var notes: [String] = [
            "Vision classify proxy (not full CLIP). Install Core ML CLIP under \(defaultModelDirectory.path) for real CLIPScore.",
        ]
        if hitWeight < 0.05 {
            notes.append("Few label–prompt token overlaps; treat score as weak signal only.")
        }

        return Metrics(
            score: score,
            backend: "vision_proxy",
            available: true,
            cosine: nil,
            topLabels: Array(labels.prefix(8)),
            notes: notes
        )
    }

    private static func tokenize(_ prompt: String) -> Set<String> {
        let lower = prompt.lowercased()
        let parts = lower.split { !$0.isLetter }.map(String.init).filter { $0.count >= 3 }
        // Drop stopwords
        let stop: Set<String> = [
            "the", "and", "with", "from", "that", "this", "into", "over", "under",
            "soft", "light", "dark", "very", "more", "than", "same", "exact", "keep",
            "while", "keeping", "using", "onto", "around", "about", "have", "been",
        ]
        return Set(parts.filter { !stop.contains($0) })
    }
}
