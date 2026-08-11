import Foundation

/// Structured analysis result for agents and humans.
public struct ImageAnalysisReport: Sendable, Codable, Equatable {
    public var version: String
    public var imagePath: String
    public var referencePath: String?
    public var prompt: String?

    public var technical: TechnicalQuality.Metrics
    public var reference: ReferenceCompare.Metrics?
    public var promptAlignment: PromptAlignment.Metrics
    /// Filled after multimodal vision review (optional).
    public var vision: VisionReview.Assessment?

    /// Ordered findings for the agent (severity + message). Pixel + vision merged.
    public var findings: [Finding]
    /// Single 0…100 rollup (weights depend on available sections).
    public var overallScore: Float
    public var summary: String

    public struct Finding: Sendable, Codable, Equatable {
        public enum Severity: String, Sendable, Codable {
            case info
            case warn
            case fail
        }
        public var severity: Severity
        public var code: String
        public var message: String
    }
}

public enum ImageAnalysisReportBuilder {
    public static let schemaVersion = "1.1"

    public static func build(
        imagePath: String,
        referencePath: String?,
        prompt: String?,
        technical: TechnicalQuality.Metrics,
        reference: ReferenceCompare.Metrics?,
        promptAlignment: PromptAlignment.Metrics
    ) -> ImageAnalysisReport {
        var findings: [ImageAnalysisReport.Finding] = []

        // Technical gates
        // Soft product photos with bokeh can legitimately sit ~40–120; only warn if very soft.
        if technical.sharpnessLaplacianVar < 20 {
            findings.append(.init(
                severity: .warn, code: "soft_focus",
                message: String(
                    format: "Low sharpness (laplacian var=%.1f). Image may be blurry or over-smoothed.",
                    technical.sharpnessLaplacianVar
                )
            ))
        }
        if technical.clipWhiteFraction > 0.08 {
            findings.append(.init(
                severity: .warn, code: "highlight_clip",
                message: String(
                    format: "Highlight clipping: %.1f%% near-white pixels.",
                    technical.clipWhiteFraction * 100
                )
            ))
        }
        if technical.clipBlackFraction > 0.12 {
            findings.append(.init(
                severity: .info, code: "shadow_crush",
                message: String(
                    format: "Deep shadows: %.1f%% near-black pixels.",
                    technical.clipBlackFraction * 100
                )
            ))
        }
        if technical.stdLuminance < 0.06 {
            findings.append(.init(
                severity: .warn, code: "low_contrast",
                message: String(format: "Low luminance contrast (σ=%.3f).", technical.stdLuminance)
            ))
        }
        if technical.noiseProxy > 0.05 {
            findings.append(.init(
                severity: .info, code: "grain",
                message: String(format: "Elevated high-frequency residual (%.4f) — grain or compression.", technical.noiseProxy)
            ))
        }
        if technical.width % 16 != 0 || technical.height % 16 != 0 {
            findings.append(.init(
                severity: .info, code: "non_aligned_dims",
                message: "Dimensions \(technical.width)×\(technical.height) not multiples of 16 (Aestrix canvas prefer 16)."
            ))
        }

        // Prompt color
        if let match = promptAlignment.colorMatch, match == false {
            findings.append(.init(
                severity: .fail, code: "color_mismatch",
                message: "Prompt colors \(promptAlignment.requestedColors) but dominant hue is '\(promptAlignment.imageDominantHue)'."
            ))
        }
        if promptAlignment.colorMatch == true {
            findings.append(.init(
                severity: .info, code: "color_match",
                message: "Prompt color intent \(promptAlignment.requestedColors) found in image (dominant='\(promptAlignment.imageDominantHue)')."
            ))
        }
        for note in promptAlignment.notes where note.contains("Short prompt") {
            findings.append(.init(severity: .info, code: "prompt_short", message: note))
        }
        if !promptAlignment.unverifiableKeywords.isEmpty {
            findings.append(.init(
                severity: .info, code: "needs_visual_review",
                message: "Unverifiable without VLM/human: \(promptAlignment.unverifiableKeywords.joined(separator: ", "))."
            ))
        }

        // Reference
        if let ref = reference {
            if ref.ssim < 0.35 {
                findings.append(.init(
                    severity: .info, code: "low_structure_fidelity",
                    message: String(
                        format: "SSIM=%.3f vs reference — large structural change (expected at high I2I strength).",
                        ref.ssim
                    )
                ))
            } else if ref.ssim > 0.85 {
                findings.append(.init(
                    severity: .info, code: "high_structure_fidelity",
                    message: String(format: "SSIM=%.3f — composition closely matches reference.", ref.ssim)
                ))
            }
            if ref.meanDeltaE > 25 {
                findings.append(.init(
                    severity: .info, code: "global_recolor",
                    message: String(format: "Mean ΔE≈%.1f — strong global color shift from reference.", ref.meanDeltaE)
                ))
            }
        }

        // Overall score
        var overall: Float
        if let ref = reference, promptAlignment.colorMatch != nil {
            // I2I with color intent: blend fidelity + color alignment + technical
            let colorPart = promptAlignment.alignmentScore
            overall = 0.35 * technical.technicalScore + 0.30 * ref.fidelityScore + 0.35 * colorPart
        } else if let ref = reference {
            overall = 0.45 * technical.technicalScore + 0.55 * ref.fidelityScore
        } else {
            overall = 0.55 * technical.technicalScore + 0.45 * promptAlignment.alignmentScore
        }
        overall = max(0, min(100, overall))

        let fails = findings.filter { $0.severity == .fail }.count
        let warns = findings.filter { $0.severity == .warn }.count
        let summary: String
        if fails > 0 {
            summary = String(
                format: "Score %.0f/100 with %d failure(s), %d warning(s). Primary issue: %@.",
                overall, fails, warns, findings.first(where: { $0.severity == .fail })?.code ?? "unknown"
            )
        } else if warns > 0 {
            summary = String(
                format: "Score %.0f/100 with %d warning(s). Technical %.0f, prompt-align %.0f%@.",
                overall, warns, technical.technicalScore, promptAlignment.alignmentScore,
                reference.map { String(format: ", fidelity %.0f", $0.fidelityScore) } ?? ""
            )
        } else {
            summary = String(
                format: "Score %.0f/100 — no hard failures. Sharpness=%.0f, hue=%@, tech=%.0f%@.",
                overall,
                technical.sharpnessLaplacianVar,
                technical.dominantHue,
                technical.technicalScore,
                reference.map { String(format: ", SSIM=%.3f", $0.ssim) } ?? ""
            )
        }

        return ImageAnalysisReport(
            version: schemaVersion,
            imagePath: imagePath,
            referencePath: referencePath,
            prompt: prompt,
            technical: technical,
            reference: reference,
            promptAlignment: promptAlignment,
            vision: nil,
            findings: findings,
            overallScore: overall,
            summary: summary
        )
    }

    /// Merge agent vision assessment into a pixel report (recomputes overall + findings).
    public static func mergingVision(
        _ report: ImageAnalysisReport,
        _ assessment: VisionReview.Assessment
    ) -> ImageAnalysisReport {
        var r = report
        r.vision = assessment
        r.version = schemaVersion

        // Prefix vision finding codes if not already vision_
        let visionFindings = assessment.findings.map { f -> ImageAnalysisReport.Finding in
            if f.code.hasPrefix("vision_") { return f }
            return ImageAnalysisReport.Finding(
                severity: f.severity, code: "vision_\(f.code)", message: f.message)
        }
        r.findings = report.findings + visionFindings

        // Blend scores: pixel overall with vision judgment.
        let pixel = report.overallScore
        let vis = assessment.visionScore
        r.overallScore = max(0, min(100, 0.45 * pixel + 0.55 * vis))

        let fails = r.findings.filter { $0.severity == .fail }.count
        let warns = r.findings.filter { $0.severity == .warn }.count
        if fails > 0 {
            r.summary = String(
                format: "Score %.0f/100 (pixel %.0f + vision %.0f). %d fail(s), %d warn(s). Vision: %@",
                r.overallScore, pixel, vis, fails, warns, assessment.verdict
            )
        } else {
            r.summary = String(
                format: "Score %.0f/100 (pixel %.0f + vision %.0f). Vision: %@ — %@",
                r.overallScore, pixel, vis, assessment.verdict, assessment.caption
            )
        }
        return r
    }

    public static func jsonString(_ report: ImageAnalysisReport, pretty: Bool = true) throws -> String {
        let enc = JSONEncoder()
        if pretty {
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        }
        enc.keyEncodingStrategy = .convertToSnakeCase
        let data = try enc.encode(report)
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    public static func textSummary(_ report: ImageAnalysisReport) -> String {
        var lines: [String] = []
        lines.append("Aestrix Image Analysis v\(report.version)")
        lines.append("image: \(report.imagePath)")
        if let r = report.referencePath { lines.append("reference: \(r)") }
        if let p = report.prompt { lines.append("prompt: \(p)") }
        lines.append(String(format: "overall: %.1f / 100", report.overallScore))
        lines.append("summary: \(report.summary)")
        lines.append("")
        lines.append("--- technical ---")
        let t = report.technical
        lines.append(String(
            format: "size=%dx%d  sharp_var=%.1f  contrast=%.3f  sat=%.3f  entropy=%.2f  score=%.1f",
            t.width, t.height, t.sharpnessLaplacianVar, t.stdLuminance, t.meanSaturation,
            t.luminanceEntropy, t.technicalScore
        ))
        lines.append(String(
            format: "mean_rgb=[%.3f, %.3f, %.3f]  hue=%@ (%.0f%%)  chromatic=%@  clip_black=%.2f%%  clip_white=%.2f%%",
            t.meanRGB[0], t.meanRGB[1], t.meanRGB[2],
            t.dominantHue, t.dominantHueFraction * 100,
            t.topChromaticHues.prefix(4).joined(separator: ","),
            t.clipBlackFraction * 100, t.clipWhiteFraction * 100
        ))
        if let ref = report.reference {
            lines.append("")
            lines.append("--- vs reference ---")
            lines.append(String(
                format: "ssim=%.4f  psnr=%.1f  mae=%.4f  hist_corr=%.3f  deltaE≈%.1f  fidelity=%.1f",
                ref.ssim, ref.psnr, ref.meanAbsError, ref.histogramCorrelation,
                ref.meanDeltaE, ref.fidelityScore
            ))
        }
        lines.append("")
        lines.append("--- prompt alignment ---")
        let pa = report.promptAlignment
        lines.append(String(
            format: "words=%d  style=%@  colors=%@  color_match=%@  score=%.1f",
            pa.promptWordCount, pa.promptStyleHint, pa.requestedColors.description,
            pa.colorMatch.map { $0 ? "yes" : "no" } ?? "n/a",
            pa.alignmentScore
        ))
        if let v = report.vision {
            lines.append("")
            lines.append("--- vision ---")
            lines.append(String(format: "score=%.1f  verdict=%@", v.visionScore, v.verdict))
            lines.append("caption: \(v.caption)")
            for (k, ans) in v.answers.sorted(by: { $0.key < $1.key }) where !ans.isEmpty {
                lines.append("  \(k): \(ans)")
            }
        }
        if !report.findings.isEmpty {
            lines.append("")
            lines.append("--- findings ---")
            for f in report.findings {
                lines.append("[\(f.severity.rawValue.uppercased())] \(f.code): \(f.message)")
            }
        }
        return lines.joined(separator: "\n")
    }
}
