import Foundation

/// Lightweight prompt ↔ image heuristics (no VLM). Complements technical/ref metrics.
public enum PromptAlignment {
    public struct Metrics: Sendable, Codable, Equatable {
        /// Color words found in prompt.
        public var requestedColors: [String]
        /// Whether dominant image hue matches any requested color (or synonym).
        public var colorMatch: Bool?
        public var imageDominantHue: String
        public var imageDominantHueFraction: Float
        /// Prompt length (words) vs BFL klein guidance (~40–70 ideal narrative).
        public var promptWordCount: Int
        public var promptStyleHint: String
        /// Keywords from prompt that are hard to verify without a VLM (listed for agent).
        public var unverifiableKeywords: [String]
        /// 0…100 heuristic alignment score (color + style only).
        public var alignmentScore: Float
        public var notes: [String]
    }

    private static let colorLexicon: [String: [String]] = [
        "red": ["red", "crimson", "scarlet", "ruby", "cherry"],
        "orange": ["orange", "terracotta", "amber", "rust"],
        "yellow": ["yellow", "gold", "golden", "lemon"],
        "green": ["green", "emerald", "olive", "lime", "forest"],
        "cyan": ["cyan", "teal", "turquoise", "aqua"],
        "blue": ["blue", "cobalt", "navy", "azure", "indigo", "cerulean"],
        "purple": ["purple", "violet", "lavender", "magenta", "pink"],
        "neutral": ["white", "black", "gray", "grey", "beige", "cream", "linen"],
    ]

    public static func analyze(prompt: String?, technical: TechnicalQuality.Metrics) -> Metrics {
        guard let prompt, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return Metrics(
                requestedColors: [],
                colorMatch: nil,
                imageDominantHue: technical.dominantHue,
                imageDominantHueFraction: technical.dominantHueFraction,
                promptWordCount: 0,
                promptStyleHint: "no_prompt",
                unverifiableKeywords: [],
                alignmentScore: 50,
                notes: ["No prompt provided; color/subject alignment skipped."]
            )
        }

        let lower = prompt.lowercased()
        let words = lower.split { !$0.isLetter && $0 != "#" }.map(String.init)
        let wordSet = Set(words)
        let wordCount = words.count

        var requested: [String] = []
        for (bucket, synonyms) in colorLexicon {
            for s in synonyms where wordSet.contains(s) {
                if !requested.contains(bucket) { requested.append(bucket) }
            }
        }
        let requestedHex = parseHexColors(lower)
        for hex in requestedHex where !requested.contains(hex.bucket) {
            requested.append(hex.bucket)
        }

        var notes: [String] = []
        var colorMatch: Bool?
        if !requested.isEmpty {
            let img = technical.dominantHue
            let chromatic = technical.topChromaticHues
            let weights = technical.hueWeights

            // Present if dominant, top-2 chromatic, or ≥12% of weighted mass (subject on large bg).
            func present(_ bucket: String) -> Bool {
                if img == bucket { return true }
                if chromatic.prefix(2).contains(bucket) { return true }
                if (weights[bucket] ?? 0) >= 0.12 { return true }
                if bucket == "purple", chromatic.prefix(2).contains("magenta") { return true }
                if bucket == "blue", chromatic.prefix(2).contains("cyan") { return true }
                return false
            }

            let mean = technical.meanRGB
            let hexMatch = requestedHex.contains { hex in
                guard mean.count == 3 else { return false }
                let dr = mean[0] - hex.r, dg = mean[1] - hex.g, db = mean[2] - hex.b
                return sqrt(dr * dr + dg * dg + db * db) <= 0.38 || present(hex.bucket)
            }
            colorMatch = requested.contains(where: present) || hexMatch

            if colorMatch == true {
                let hit = requested.first(where: present) ?? img
                notes.append(
                    "Requested color present (hit='\(hit)'; dominant='\(img)' \(String(format: "%.0f", technical.dominantHueFraction * 100))%; chromatic top=\(chromatic.prefix(3).joined(separator: ",")))."
                )
            } else {
                notes.append(
                    "Prompt colors \(requested) not found in image hues (dominant='\(img)', chromatic=\(chromatic.prefix(4).joined(separator: ","))). Raise I2I strength or rephrase color."
                )
            }
        } else {
            notes.append("No explicit color words in prompt; color match not scored.")
        }

        let style: String
        if wordCount < 12 {
            style = "short_prompt"
            notes.append("Short prompt (\(wordCount) words); Klein often prefers ~40–70 word narrative.")
        } else if wordCount <= 80 {
            style = "narrative_ok"
        } else {
            style = "long_prompt"
            notes.append("Long prompt (\(wordCount) words); may dilute subject.")
        }

        // Keywords that need VLM / human eyes
        let hard = [
            "text", "typography", "logo", "face", "hands", "fingers", "anatomy",
            "reading", "sign", "watermark", "blurry", "artifact",
        ]
        let foundHard = hard.filter { wordSet.contains($0) }
        if !foundHard.isEmpty {
            notes.append(
                "Prompt mentions \(foundHard) — verify visually/VLM; pixel metrics cannot judge these."
            )
        }

        var score: Float = 60
        if let cm = colorMatch {
            score = cm ? 85 : 25
            // Weight by how dominant the hue is
            if cm { score += min(15, technical.dominantHueFraction * 30) }
        }
        if style == "narrative_ok" { score = min(100, score + 5) }
        if style == "short_prompt" { score = max(0, score - 5) }

        return Metrics(
            requestedColors: requested,
            colorMatch: colorMatch,
            imageDominantHue: technical.dominantHue,
            imageDominantHueFraction: technical.dominantHueFraction,
            promptWordCount: wordCount,
            promptStyleHint: style,
            unverifiableKeywords: foundHard,
            alignmentScore: max(0, min(100, score)),
            notes: notes
        )
    }

    private struct HexColor {
        var r: Float
        var g: Float
        var b: Float
        var bucket: String
    }

    private static func parseHexColors(_ text: String) -> [HexColor] {
        guard let regex = try? NSRegularExpression(pattern: #"(?<![0-9a-f])#([0-9a-f]{6})(?![0-9a-f])"#) else {
            return []
        }
        let nsRange = NSRange(text.startIndex ..< text.endIndex, in: text)
        return regex.matches(in: text, range: nsRange).compactMap { match in
            guard let digitsRange = Range(match.range(at: 1), in: text),
                  let value = UInt32(text[digitsRange], radix: 16)
            else { return nil }
            let r = Float((value >> 16) & 0xff) / 255
            let g = Float((value >> 8) & 0xff) / 255
            let b = Float(value & 0xff) / 255
            return HexColor(r: r, g: g, b: b, bucket: hueBucket(r: r, g: g, b: b))
        }
    }

    private static func hueBucket(r: Float, g: Float, b: Float) -> String {
        let maxV = max(r, max(g, b)), minV = min(r, min(g, b))
        let delta = maxV - minV
        guard maxV > 0.12, delta / maxV >= 0.18 else { return "neutral" }
        var hue: Float
        if maxV == r { hue = 60 * ((g - b) / delta).truncatingRemainder(dividingBy: 6) }
        else if maxV == g { hue = 60 * ((b - r) / delta + 2) }
        else { hue = 60 * ((r - g) / delta + 4) }
        if hue < 0 { hue += 360 }
        if hue < 15 || hue >= 345 { return "red" }
        if hue < 45 { return "orange" }
        if hue < 70 { return "yellow" }
        if hue < 160 { return "green" }
        if hue < 200 { return "cyan" }
        if hue < 260 { return "blue" }
        return "purple"
    }
}
