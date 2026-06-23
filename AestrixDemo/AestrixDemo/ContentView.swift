import SwiftUI
import AestrixEngine

/// Minimal demo that proves the MLX dependency resolves and executes on-device.
/// (Milestone 5 will wire real text→image generation here.)
struct ContentView: View {
    @State private var report: String = "Tap to verify the MLX integration."
    @State private var busy = false

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "sparkles")
                .font(.system(size: 48))
                .foregroundStyle(.tint)

            Text("AestrixEngine")
                .font(.largeTitle.bold())
            Text("FLUX.2-klein-4B · MLX on iOS")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Button {
                runCheck()
            } label: {
                Label("Run MLX integration check", systemImage: "bolt.fill")
            }
            .buttonStyle(.borderedProminent)
            .disabled(busy)

            ScrollView {
                Text(report)
                    .font(.system(.body, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding()
    }

    private func runCheck() {
        busy = true
        report = "Running a 512×512 matmul + sum on-device…"
        let started = Date()
        Task { [started] in
            let engine = AestrixEngine()
            let r = await engine.verifyIntegration()
            let elapsed = Date().timeIntervalSince(started)
            await MainActor.run {
                report = """
                ok      : \(r.ok ? "YES" : "NO")
                matmul  : \(r.matmulRows)×\(r.matmulRows)
                sum     : \(r.resultSum)
                elapsed : \(String(format: "%.3f", elapsed)) s
                \(r.message)
                """
                busy = false
            }
        }
    }
}
