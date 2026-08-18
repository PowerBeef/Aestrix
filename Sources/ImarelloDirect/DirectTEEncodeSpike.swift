import Foundation
import MLX
import ImarelloText
import ImarelloWeights
import ImarelloCore

/// Milestone D verification: the complete direct encode against the REAL
/// production TextEncoderModule (bf16 path), on a real chat-templated prompt.
public enum DirectTEEncodeSpike {

    public static func run(
        snapshot: ModelSnapshot, metallibURL: URL, prompt: String
    ) async throws -> String {
        let t0 = CFAbsoluteTimeGetCurrent()
        let direct = try DirectTEEncoder(
            teDirectory: snapshot.textEncoderDirectory,
            tokenizerDirectory: snapshot.tokenizerDirectory,
            metallibURL: metallibURL)
        let buildMS = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        // Embedding dequant sanity on a few tokens.
        var embedLines: [String] = []
        for tok in [0, 1_000, 77_777, 151_000] {
            let (cos, maxAbs) = direct.verifyEmbedding(token: tok)
            embedLines.append(String(
                format: "  embed[%6d]: cosine=%.7f maxAbs=%.5f", tok, cos, maxAbs))
        }

        // Direct encode (cold then warm).
        _ = try direct.encode(prompt)
        let (dEmbeds, realTokens, _) = try direct.encode(prompt)
        var directMS = 0.0
        do {
            let t = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< 3 { _ = try direct.encode(prompt) }
            directMS = (CFAbsoluteTimeGetCurrent() - t) * 1000 / 3
        }

        // Real production TE (bf16), resident, warm.
        let te = TextEncoderModule(snapshot: snapshot)
        try await te.load()
        let (rEmbeds, rReal) = try te.encode(prompt)
        var realMS = 0.0
        do {
            _ = try te.encode(prompt)
            let t = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< 3 { _ = try te.encode(prompt) }
            realMS = (CFAbsoluteTimeGetCurrent() - t) * 1000 / 3
        }

        // Compare: overall, real rows, pad rows.
        let L = 512, dim = 7680
        let d = dEmbeds.reshaped([L, dim]).asType(.float32).asArray(Float.self)
        let r = rEmbeds.reshaped([L, dim]).asType(.float32).asArray(Float.self)
        let dNaN = d.lazy.filter { !$0.isFinite }.count
        let rMax = r.lazy.map { abs($0) }.max() ?? 0
        FileHandle.standardError.write(Data(
            "debug: d_nonfinite=\(dNaN)/\(d.count) real_max_abs=\(rMax)\n".utf8))
        func cosine(_ rows: Range<Int>) -> (Double, Double) {
            var dot = 0.0, na = 0.0, nb = 0.0, maxDiff = 0.0
            for row in rows {
                for c in 0 ..< dim {
                    let a = Double(d[row * dim + c]), b = Double(r[row * dim + c])
                    dot += a * b; na += a * a; nb += b * b
                    maxDiff = max(maxDiff, abs(a - b))
                }
            }
            return (dot / (na.squareRoot() * nb.squareRoot() + 1e-30), maxDiff)
        }
        let (cosAll, maxAll) = cosine(0 ..< L)
        let (cosReal, maxReal) = cosine(0 ..< realTokens)
        let (cosPad, maxPad) = cosine(realTokens ..< L)

        return """
        direct-te-encode (milestone D) — real prompt, chat template, production mask semantics
          prompt_tokens:    \(realTokens) real (module says \(rReal)) of \(L)
        \(embedLines.joined(separator: "\n"))
          vs REAL TE (bf16 product path):
            all 512 rows:   cosine=\(String(format: "%.7f", cosAll)) maxAbs=\(String(format: "%.4f", maxAll))
            real rows:      cosine=\(String(format: "%.7f", cosReal)) maxAbs=\(String(format: "%.4f", maxReal))
            pad rows:       cosine=\(String(format: "%.7f", cosPad)) maxAbs=\(String(format: "%.4f", maxPad))
          engine_build:     \(String(format: "%.0f", buildMS)) ms (one-time: weights+tokenizer+PSOs)
          direct_encode:    \(String(format: "%.1f", directMS)) ms (tokenize + CPU embed + 27 layers + taps)
          real_te_encode:   \(String(format: "%.1f", realMS)) ms (resident, warm, bf16)
          speedup_resident: \(String(format: "%.2f", realMS / directMS))×
        """
    }
}
