import Foundation

/// Multimodal vision layer for agents that can **see** images (e.g. read PNG via vision tools).
///
/// Pixel metrics cannot judge text legibility, anatomy, “is it really a mug”, or aesthetic quality.
/// This module defines a **checklist + answer schema** the agent fills after inspecting the image,
/// then merges into ``ImageAnalysisReport``.
public enum VisionReview {
    public enum Mode: String, Sendable, Codable, CaseIterable {
        case t2i
        case i2i
    }

    /// One checklist item for the vision pass.
    public struct Question: Sendable, Codable, Equatable {
        public var id: String
        public var prompt: String
        /// Whether a “no” / failure should become a hard gate.
        public var critical: Bool
    }

    /// Agent-filled answers after viewing the image.
    public struct Assessment: Sendable, Codable, Equatable {
        public var mode: Mode
        public var imagePath: String
        public var referencePath: String?
        public var prompt: String?

        /// Free-form caption of what the image actually depicts.
        public var caption: String
        /// Answers keyed by ``Question/id``.
        public var answers: [String: String]
        /// Agent vision findings (merged into report findings).
        public var findings: [ImageAnalysisReport.Finding]
        /// Subjective / semantic score 0…100 (agent judgment).
        public var visionScore: Float
        /// Short overall judgment for the agent log.
        public var verdict: String

        public init(
            mode: Mode,
            imagePath: String,
            referencePath: String? = nil,
            prompt: String? = nil,
            caption: String,
            answers: [String: String] = [:],
            findings: [ImageAnalysisReport.Finding] = [],
            visionScore: Float,
            verdict: String
        ) {
            self.mode = mode
            self.imagePath = imagePath
            self.referencePath = referencePath
            self.prompt = prompt
            self.caption = caption
            self.answers = answers
            self.findings = findings
            self.visionScore = max(0, min(100, visionScore))
            self.verdict = verdict
        }
    }

    // MARK: - Checklist

    public static func checklist(mode: Mode) -> [Question] {
        var q: [Question] = [
            .init(id: "subject", prompt: "What is the main subject? Does it match the prompt?", critical: true),
            .init(id: "color", prompt: "Do primary colors match the prompt (including product/subject color)?", critical: true),
            .init(id: "lighting", prompt: "Is lighting coherent and consistent with the prompt?", critical: false),
            .init(id: "artifacts", prompt: "Any generative artifacts (melting edges, extra limbs, weird reflections)?", critical: true),
            .init(id: "composition", prompt: "Is framing/composition intentional (not cut off awkwardly)?", critical: false),
            .init(id: "text", prompt: "If text/logo/signage was requested: is it legible and correct? (else N/A)", critical: true),
            .init(id: "anatomy", prompt: "If people/hands/animals: anatomy OK? (else N/A)", critical: true),
            .init(id: "photoreal", prompt: "Material quality (ceramic, metal, skin, fabric) look plausible?", critical: false),
        ]
        if mode == .i2i {
            q.append(contentsOf: [
                .init(
                    id: "identity",
                    prompt: "Does the result preserve the reference identity (pose/layout) where intended?",
                    critical: false
                ),
                .init(
                    id: "edit_applied",
                    prompt: "Was the requested edit actually applied (not just a re-render of the source)?",
                    critical: true
                ),
            ])
        }
        return q
    }

    /// Markdown brief for the agent: metrics summary + checklist + paths to open with vision.
    public static func agentBrief(report: ImageAnalysisReport, mode: Mode) -> String {
        var lines: [String] = []
        lines.append("# Vision review brief (Imarello)")
        lines.append("")
        lines.append("## Paths (open with vision / read image tool)")
        lines.append("- **generated:** `\(report.imagePath)`")
        if let r = report.referencePath {
            lines.append("- **reference:** `\(r)`")
        }
        if let p = report.prompt {
            lines.append("- **prompt:** \(p)")
        }
        lines.append("")
        lines.append("## Pixel harness (already computed)")
        lines.append(String(format: "- overall_score: %.1f", report.overallScore))
        lines.append(String(format: "- technical_score: %.1f  sharp=%.1f  hue=%@ chromatic=%@",
            report.technical.technicalScore,
            report.technical.sharpnessLaplacianVar,
            report.technical.dominantHue,
            report.technical.topChromaticHues.prefix(4).joined(separator: ",")))
        if let ref = report.reference {
            lines.append(String(format: "- vs_ref: SSIM=%.3f fidelity=%.1f ΔE≈%.1f",
                ref.ssim, ref.fidelityScore, ref.meanDeltaE))
        }
        lines.append(String(format: "- prompt_align: score=%.1f color_match=%@",
            report.promptAlignment.alignmentScore,
            report.promptAlignment.colorMatch.map { $0 ? "yes" : "no" } ?? "n/a"))
        lines.append("- pixel_summary: \(report.summary)")
        if !report.findings.isEmpty {
            lines.append("- pixel_findings:")
            for f in report.findings {
                lines.append("  - [\(f.severity.rawValue)] \(f.code): \(f.message)")
            }
        }
        lines.append("")
        lines.append("## Gaps only vision can fill")
        lines.append("- Subject identity & prompt adherence beyond color buckets")
        lines.append("- Text/logo readability, hands/anatomy, extra fingers, melted geometry")
        lines.append("- Aesthetic quality, material realism, lighting logic")
        if mode == .i2i {
            lines.append("- Whether the edit actually landed while preserving intended identity")
        }
        lines.append("")
        lines.append("## Checklist — answer each after viewing the image")
        for q in checklist(mode: mode) {
            let crit = q.critical ? " **[critical]**" : ""
            lines.append("- **\(q.id)**\(crit): \(q.prompt)")
        }
        lines.append("")
        lines.append("## Output format (fill Assessment JSON fields)")
        lines.append("```")
        lines.append("caption: <one sentence>")
        lines.append("answers: { subject, color, lighting, artifacts, ... }")
        lines.append("findings: [{ severity, code: vision_*, message }]")
        lines.append("vision_score: 0-100")
        lines.append("verdict: one line")
        lines.append("```")
        lines.append("")
        lines.append("Then call `ImageAnalysisReportBuilder.mergingVision(report, assessment)` or re-emit combined JSON.")
        return lines.joined(separator: "\n")
    }

    /// Build a skeleton Assessment (empty answers) for tools to fill.
    public static func emptyAssessment(
        report: ImageAnalysisReport,
        mode: Mode
    ) -> Assessment {
        var answers: [String: String] = [:]
        for q in checklist(mode: mode) {
            answers[q.id] = ""
        }
        return Assessment(
            mode: mode,
            imagePath: report.imagePath,
            referencePath: report.referencePath,
            prompt: report.prompt,
            caption: "",
            answers: answers,
            findings: [],
            visionScore: 50,
            verdict: "pending_vision_review"
        )
    }
}
