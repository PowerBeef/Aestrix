import Foundation
import AestrixCore

/// Canvas / sequence analytics for pressure reports (cheap; no Metal).
public enum PressureAnalytics {
    /// Packed spatial size after VAE 8× and 2×2 patchify → /16.
    public static func packedSpatial(width: Int, height: Int) -> (h: Int, w: Int) {
        let scale = ModelConstants.vaeScaleFactor * 2
        return (height / scale, width / scale)
    }

    public static func canvasStats(
        width: Int,
        height: Int,
        textSeqLen: Int = ModelConstants.maxSequenceLength
    ) -> CanvasAnalytics {
        let (ph, pw) = packedSpatial(width: width, height: height)
        let imageSeq = ph * pw
        let joint = textSeqLen + imageSeq
        // Rough: one f32 activation [1, L, innerDim] + similar residual (~2×).
        let act = UInt64(joint) * UInt64(ModelConstants.innerDim) * 4 * 2
        let note =
            "joint=text(\(textSeqLen))+img(\(imageSeq)); single-stream L=\(joint); "
            + "est ~\(act / 1_048_576) MiB for 2×[1,L,\(ModelConstants.innerDim)] f32 (not full attn temps)"
        return CanvasAnalytics(
            width: width,
            height: height,
            textSeqLen: textSeqLen,
            imageSeqLen: imageSeq,
            jointSeqLen: joint,
            packedH: ph,
            packedW: pw,
            estSingleStreamActBytes: act,
            note: note
        )
    }

    public static func rank(
        samples: [MemoryPoint],
        recommendedWS: UInt64?,
        top: Int = 15
    ) -> (byActive: [PressureRankedPoint], byDelta: [PressureRankedPoint]) {
        func share(_ active: UInt64) -> Double? {
            guard let ws = recommendedWS, ws > 0 else { return nil }
            return Double(active) / Double(ws)
        }
        let byActive = samples
            .sorted { $0.mlxActiveBytes > $1.mlxActiveBytes }
            .prefix(top)
            .map {
                PressureRankedPoint(
                    label: $0.label,
                    mlxActiveBytes: $0.mlxActiveBytes,
                    mlxActiveDeltaBytes: $0.mlxActiveDeltaBytes ?? 0,
                    shareOfRecommendedWS: share($0.mlxActiveBytes),
                    note: $0.note
                )
            }
        let byDelta = samples
            .sorted { ($0.mlxActiveDeltaBytes ?? 0) > ($1.mlxActiveDeltaBytes ?? 0) }
            .prefix(top)
            .map {
                PressureRankedPoint(
                    label: $0.label,
                    mlxActiveBytes: $0.mlxActiveBytes,
                    mlxActiveDeltaBytes: $0.mlxActiveDeltaBytes ?? 0,
                    shareOfRecommendedWS: share($0.mlxActiveBytes),
                    note: $0.note
                )
            }
        return (Array(byActive), Array(byDelta))
    }

    public static func phasePeaks(samples: [MemoryPoint]) -> [String: UInt64] {
        var peaks: [String: UInt64] = [:]
        for s in samples {
            let phase: String
            if s.label.hasPrefix("te.") || s.label.contains("load_te") || s.label.contains("encode_te")
                || s.label.contains("_te")
            {
                phase = "te"
            } else if s.label.hasPrefix("dit.") || s.label.contains("denoise") || s.label.contains("load_dit")
                || s.label.contains("rope")
            {
                phase = "dit"
            } else if s.label.hasPrefix("vae.") || s.label.contains("vae") || s.label.contains("decode")
            {
                phase = "vae"
            } else {
                phase = "other"
            }
            peaks[phase] = max(peaks[phase] ?? 0, s.mlxActiveBytes)
        }
        return peaks
    }
}
