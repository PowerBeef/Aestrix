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
    /// CLIP / Vision semantic prompt–image score (P1).
    public var semantic: SemanticAlignment.Metrics?
    /// I2I strength used when generating (P2 strength-aware gates).
    public var i2iStrength: Float?
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
    /// Schema 1.4: unstructured-garbage hard fail (TV-static / f16 overflow).
    public static let schemaVersion = "1.4"

    public static func build(
        imagePath: String,
        referencePath: String?,
        prompt: String?,
        technical: TechnicalQuality.Metrics,
        reference: ReferenceCompare.Metrics?,
        promptAlignment: PromptAlignment.Metrics,
        semantic: SemanticAlignment.Metrics? = nil,
        i2iStrength: Float? = nil
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
        // Imarello VAE auto-tiles when unpatchified spatial ≥ 96 (~768 px). Flag hard seams.
        if technical.expectsVAETiling, technical.tileSeamScore > 3.0 {
            findings.append(.init(
                severity: .warn, code: "possible_tile_seam",
                message: String(
                    format: "Elevated tile-seam score=%.2f (V=%.2f H=%.2f) on ≥768 canvas — check cosine VAE blend / hard 2×2 seams.",
                    technical.tileSeamScore, technical.tileSeamVertical, technical.tileSeamHorizontal
                )
            ))
        } else if technical.expectsVAETiling {
            findings.append(.init(
                severity: .info, code: "vae_tile_expected",
                message: String(
                    format: "Canvas %dx%d likely used tiled VAE decode; seam_score=%.2f (clean typically <2.5).",
                    technical.width, technical.height, technical.tileSeamScore
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
        if TechnicalQuality.looksLikeUnstructuredGarbage(technical) {
            findings.append(.init(
                severity: .fail, code: "unstructured_garbage",
                message: String(
                    format: "Unstructured speckle (noise=%.3f sharp=%.0f hue_frac=%.2f autocorr=%.2f) — decode/GEMM garbage, not a picture.",
                    technical.noiseProxy,
                    technical.sharpnessLaplacianVar,
                    technical.dominantHueFraction,
                    technical.spatialAutocorrLag2
                )
            ))
        } else if technical.noiseProxy > 0.05 {
            findings.append(.init(
                severity: .info, code: "grain",
                message: String(format: "Elevated high-frequency residual (%.4f) — grain or compression.", technical.noiseProxy)
            ))
        }
        if technical.width % 16 != 0 || technical.height % 16 != 0 {
            findings.append(.init(
                severity: .info, code: "non_aligned_dims",
                message: "Dimensions \(technical.width)×\(technical.height) not multiples of 16 (Imarello canvas prefer 16)."
            ))
        }

        // Prompt color — hard fail only for single-color intent (multi-color scenes often false-positive).
        if let match = promptAlignment.colorMatch, match == false {
            let multi = promptAlignment.requestedColors.filter { $0 != "hex" && $0 != "neutral" }.count > 1
            findings.append(.init(
                severity: multi ? .warn : .fail,
                code: "color_mismatch",
                message: "Prompt colors \(promptAlignment.requestedColors) not found (dominant='\(promptAlignment.imageDominantHue)'\(multi ? "; multi-color intent — verify with vision" : ""))."
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

        // Reference + strength-aware I2I gates (P2)
        let strength = i2iStrength
        let colorEdit = promptAlignment.colorMatch != nil
            || !(promptAlignment.requestedColors.filter { $0 != "hex" && $0 != "neutral" }.isEmpty)
        if let ref = reference {
            findings.append(.init(
                severity: .info, code: "perceptual_distance",
                message: String(
                    format: "LPIPS-lite distance=%.3f (score=%.0f, msSSIM=%.3f).",
                    ref.perceptualDistance, ref.perceptualScore, ref.msSSIM
                )
            ))

            let highStrength = (strength ?? 0) >= 0.75
            let lowStrength = (strength ?? 1) < 0.4 && strength != nil

            if ref.ssim < 0.35 {
                if highStrength {
                    findings.append(.init(
                        severity: .info, code: "expected_structure_change",
                        message: String(
                            format: "SSIM=%.3f with strength≥0.75 — large change expected for strong I2I edit.",
                            ref.ssim
                        )
                    ))
                } else {
                    findings.append(.init(
                        severity: .info, code: "low_structure_fidelity",
                        message: String(
                            format: "SSIM=%.3f vs reference — large structural change%@.",
                            ref.ssim,
                            strength.map { String(format: " (strength=%.2f)", $0) } ?? ""
                        )
                    ))
                }
            } else if ref.ssim > 0.85 {
                if highStrength && colorEdit {
                    findings.append(.init(
                        severity: .warn, code: "strength_too_low_for_edit",
                        message: String(
                            format: "SSIM=%.3f still very high with strength≥0.75 and color intent — edit may not have applied; raise strength or strengthen prompt.",
                            ref.ssim
                        )
                    ))
                } else {
                    findings.append(.init(
                        severity: .info, code: "high_structure_fidelity",
                        message: String(format: "SSIM=%.3f — composition closely matches reference.", ref.ssim)
                    ))
                }
            }

            if lowStrength && ref.ssim < 0.5 {
                findings.append(.init(
                    severity: .warn, code: "unexpected_identity_drift",
                    message: String(
                        format: "SSIM=%.3f with low strength (%.2f) — more drift than expected; check seed/pipeline.",
                        ref.ssim, strength ?? 0
                    )
                ))
            }

            // Perceptual distance gate (LPIPS-lite)
            if highStrength && colorEdit && ref.perceptualDistance < 0.08 {
                findings.append(.init(
                    severity: .warn, code: "low_perceptual_change",
                    message: String(
                        format: "LPIPS-lite distance=%.3f very small despite strength≥0.75 + color edit.",
                        ref.perceptualDistance
                    )
                ))
            }

            if ref.meanDeltaE > 25 {
                findings.append(.init(
                    severity: .info, code: "global_recolor",
                    message: String(format: "Mean ΔE≈%.1f — strong global color shift from reference.", ref.meanDeltaE)
                ))
            }
        }

        // Semantic / CLIP (P1)
        if let sem = semantic {
            if sem.available {
                findings.append(.init(
                    severity: .info, code: "semantic_score",
                    message: String(
                        format: "Semantic alignment %.0f/100 via %@%@",
                        sem.score, sem.backend,
                        sem.topLabels.isEmpty ? "" : " labels=[\(sem.topLabels.prefix(4).joined(separator: ","))]"
                    )
                ))
                if sem.score < 30 && sem.backend != "unavailable" {
                    findings.append(.init(
                        severity: .warn, code: "low_semantic_alignment",
                        message: String(
                            format: "Low prompt–image semantic score %.0f (%@). Verify subject with vision.",
                            sem.score, sem.backend
                        )
                    ))
                }
            } else {
                findings.append(.init(
                    severity: .info, code: "semantic_unavailable",
                    message: sem.notes.first ?? "Semantic alignment unavailable."
                ))
            }
        }

        // Overall score
        var overall: Float
        let semPart = semantic?.available == true ? semantic!.score : nil
        if let ref = reference, promptAlignment.colorMatch != nil {
            let colorPart = promptAlignment.alignmentScore
            if let s = semPart {
                overall = 0.25 * technical.technicalScore + 0.25 * ref.fidelityScore
                    + 0.25 * colorPart + 0.25 * s
            } else {
                overall = 0.35 * technical.technicalScore + 0.30 * ref.fidelityScore + 0.35 * colorPart
            }
        } else if let ref = reference {
            if let s = semPart {
                overall = 0.35 * technical.technicalScore + 0.40 * ref.fidelityScore + 0.25 * s
            } else {
                overall = 0.45 * technical.technicalScore + 0.55 * ref.fidelityScore
            }
        } else if let s = semPart {
            overall = 0.40 * technical.technicalScore + 0.30 * promptAlignment.alignmentScore + 0.30 * s
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
                format: "Score %.0f/100 with %d warning(s). Technical %.0f, prompt-align %.0f%@%@.",
                overall, warns, technical.technicalScore, promptAlignment.alignmentScore,
                reference.map { String(format: ", fidelity %.0f", $0.fidelityScore) } ?? "",
                semPart.map { String(format: ", semantic %.0f", $0) } ?? ""
            )
        } else {
            summary = String(
                format: "Score %.0f/100 — no hard failures. Sharpness=%.0f, hue=%@, tech=%.0f%@%@.",
                overall,
                technical.sharpnessLaplacianVar,
                technical.dominantHue,
                technical.technicalScore,
                reference.map { String(format: ", SSIM=%.3f perc=%.0f", $0.ssim, $0.perceptualScore) } ?? "",
                semPart.map { String(format: ", semantic %.0f", $0) } ?? ""
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
            semantic: semantic,
            i2iStrength: i2iStrength,
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
        lines.append("Imarello Image Analysis v\(report.version)")
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
        if t.expectsVAETiling {
            lines.append(String(
                format: "vae_tile: expected=yes  seam_score=%.2f  V=%.2f  H=%.2f",
                t.tileSeamScore, t.tileSeamVertical, t.tileSeamHorizontal
            ))
        }
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
            lines.append(String(
                format: "lpips_lite: distance=%.3f  score=%.1f  ms_ssim=%.3f",
                ref.perceptualDistance, ref.perceptualScore, ref.msSSIM
            ))
        }
        if let s = report.i2iStrength {
            lines.append(String(format: "i2i_strength: %.2f", s))
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
        if let sem = report.semantic {
            lines.append("")
            lines.append("--- semantic (CLIP / proxy) ---")
            lines.append(String(
                format: "backend=%@  score=%.1f  available=%@%@",
                sem.backend, sem.score, sem.available ? "yes" : "no",
                sem.cosine.map { String(format: "  cosine=%.4f", $0) } ?? ""
            ))
            if !sem.topLabels.isEmpty {
                lines.append("labels: \(sem.topLabels.prefix(6).joined(separator: ", "))")
            }
        }
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
