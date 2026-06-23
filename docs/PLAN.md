# AestrixEngine — Optimized MLX Engine for FLUX.2-klein-4B on iOS

## Context

**Goal:** Build a highly optimized MLX inference engine that runs
[`mlx-community/flux2-klein-4b-4bit`](https://hf.co/mlx-community/flux2-klein-4b-4bit)
on iOS at high speed with minimal RAM, supporting **text-to-image generation and
single-image instruction editing**. Target: **iPhone 15 Pro** (A17 Pro, 8 GB unified
memory), **iOS 26** minimum (newest Metal stack).

**Why this is feasible (verified, not assumed):**
- **FLUX.2-klein-4B** is a 4B-parameter *rectified-flow transformer* that **unifies
  generation + editing in one architecture**. It is distilled: the official Diffusers
  recipe runs **4 inference steps at guidance=1.0 (no CFG)** → only ~4 transformer
  forward passes per image. Apache-2.0 (commercially usable).
- **MLX-Swift officially supports iOS** — `Package.swift` declares `.iOS(.v17)`,
  links Metal + Accelerate, ships `MLX/MLXNN/MLXFast/MLXOptimizers`. Apple's WWDC25
  Session 298 ("Explore LLMs on Apple Silicon with MLX") demonstrates on-device Swift
  inference with the `ModelContainer` actor pattern and built-in quantization.
- **Weights are 4-bit MLX format** (already quantized by mlx-community): text encoder
  2.26 GB + transformer 2.18 GB + VAE 0.17 GB ≈ **4.6 GB total**. Fits 8 GB *only* with
  staged loading.
- **Canonical low-memory pattern exists** in `ml-explore/mlx-examples/flux`: *"unload
  T5/clip after conditioning, before the diffusion transformer"* + `del flow` before
  VAE decode. We generalize this for iOS wired-memory.

**Caveat carried forward:** Apple positions MLX as *research-grade, not production*.
We treat MLX-Swift as the inference core and wrap it in production-grade Swift
(error handling, memory pressure, lifecycle). Pin a specific MLX-Swift version.

**No Swift FLUX implementation exists today** — `mlx-swift-examples` has none. This is
a genuine port, sourced from Python references: `filipstrand/mflux` (the canonical MLX
FLUX port), `ml-explore/mlx-examples/flux` (Apple's FLUX.1 reference), `IonDen/mlx-taef`
(low-mem VAE decode), `IonDen/mlx-teacache` (step-skipping). The architecture bible for
FLUX.2 Klein mechanics is the Geronimo "How Inference Works" deep-dive.

---

## Architecture: the pipeline

End-to-end FLUX.2 Klein inference (text-to-image), all in MLX-Swift:

```
prompt ─▶ Qwen3 tokenizer (+chat template) ─▶ Qwen3-4B [extract layers 9,18,27]
          ─▶ ctx (B,512,7680)   +   txt_ids (512,4) = (t=0,h=0,w=0,l=0..511)

noise (128,H/16,W/16) ─flatten─▶ img (N,128)  +  img_ids (N,4) = (t=0,h,w,l=0)

  [EDIT] reference image ─▶ VAE.encode ─▶ ref latent ─flatten─▶ ref (M,128)
        + ref_ids (M,4) with t=10   ─▶ x = cat([img,ref]); x_ids = cat([img_ids,ref_ids])

schedule = linspace(1,0, steps+1) ** (1/5)        # Karras power schedule, steps=4
for (t_curr, t_next) in schedule:
    pred = Transformer(x, x_ids, t_curr, ctx, ctx_ids, guidance=1.0)
    img  = img + (t_next - t_curr) * pred
    [EDIT] keep only first img tokens of pred       # pred = pred[:, :img.count]

img ─▶ unflatten (1,128,H/16,W/16) ─▶ VAE.decode ─▶ pixels[-1,1] ─▶ clamp ─▶ UIImage
```

**Editing** = the *same* transformer + same loop; reference-image latents are simply
**concatenated** to the noise tokens and distinguished by the RoPE `t` coordinate
(t=10, 20, …). Text tokens are unchanged. No separate edit model, no mask, no inpaint
pipeline — natural-language instruction editing ("turn this cat into a dog").

---

## Component → MLX-Swift module mapping

| Component | Source of truth | MLX-Swift module |
|---|---|---|
| **Qwen3-4B text encoder** (4-bit, extract layers 9/18/27 → concat to 7680-d) | `mlx-community` `text_encoder/` + mlx-swift-examples Qwen3 loader | `Qwen3TextEncoder` (reuse mlx-swift-examples Qwen3 + quantized loader; hook hidden states) |
| **VAE** (CNN encoder + 1 attention + pixel-unshuffle; 16× spatial, 128-ch latent) | `mlx-community` `vae/` + `mflux/flux_vae` / `mlx-examples/flux` | `FluxVAE : Module` (conv stack + attention block + unshuffle) |
| **Klein transformer** (time/vec/guidance/img-patch embeds, double_blocks, single_blocks, final layer) | `mlx-community` `transformer/` + `mlx-examples/flux/flux.py` | `KleinTransformer : Module` (`@ModuleInfo` submodules) |
| **4D RoPE** `(t,h,w,l)` | Geronimo deep-dive (FLUX.2 extends FLUX.1's 3D with `t` for refs) | `RoPE4D` (custom op; FLUX.1 3D version in mlx-examples to adapt) |
| **Scheduler** (rectified-flow Euler, Karras schedule) | Geronimo deep-dive / `mflux/schedulers` | `RectifiedFlowScheduler` |
| **Orchestration + staged memory** | `mlx-examples/flux` unload pattern + mlx-swift `WiredMemory` | `Flux2KleinPipeline` (actor) |

Exact tensor dims (hidden_size, num_layers, num_heads, double/single block counts) are
read from the model's `index.json`/config at load time. Known anchors: `context_in_dim
= 7680`, `in_channels = 128`, Qwen3 has 36 layers (we extract 9/18/27, each 2560-d).

---

## Minimal-RAM strategy (the core of "minimal RAM")

8 GB iPhone 15 Pro, ~2.5 GB reserved for iOS + app → ~5–5.5 GB usable. Weights total
4.6 GB, so **resident-everything does not fit**. Solution: **staged loading** coordinated
through MLX wired-memory tickets.

```
Phase A — Condition (~2.4 GB wired):
   load Tokenizer(11 MB) + VAE(165 MB) + Qwen3-4B(2.26 GB)
   encode prompt (+ encode reference image via VAE if editing)
   mx.eval(conditioning); FREE Qwen3-4B          ← -2.26 GB

Phase B — Denoise (~3.0–4.0 GB wired):           ← peak
   load Klein Transformer(2.18 GB)
   4 forward passes (activations: ~4600-token fp16 attention)
   FREE Transformer                              ← -2.18 GB

Phase C — Decode (~0.2 GB wired):
   VAE already resident; decode; FREE VAE
```

- Peak wired ≈ **3.5–4 GB** (Phase B) + overhead → **~6–6.5 GB total < 8 GB.** ✓
- VAE stays resident throughout (cheap, used in A and C). Only the two multi-GB models
  are loaded→freed in sequence.
- **API:** `WiredSumPolicy` + `WiredMemoryTicket` (reservation for weights, active ticket
  during compute) — the modern mlx-swift API (NOT deprecated `GPU.withWiredLimit`).
- `mx.eval()` at each stage boundary forces materialization so freed arrays actually drop.
- Respond to `UIApplication.didReceiveMemoryWarning` by aborting the run and freeing all.

**Validate early on real iPhone 15 Pro hardware.** If Phase B overshoots, levers (in
order): drop default resolution 1024²→768²; fp16 activations→mixed; free Qwen3 more
aggressively; TeaCache step-skipping (fewer activations).

---

## Speed strategy ("incredible speed")

Built-in wins (already in the model design, free):
1. **4 steps** (distilled rectified flow) — minimal denoise passes.
2. **guidance = 1.0 → no CFG** — single forward per step (vs 2× for CFG models).
3. **4-bit weights** — MLX fused dequant-GEMM; ~4× smaller, faster matmuls.
4. **fp16 activations** — default compute dtype.
5. `mlx.fast.scaled_dot_product_attention` — flash attention over the ~4600-token
   sequence (non-causal).

Engine wins (Phase 7 optimization pass):
6. `compile()` the transformer forward + scheduler step — MLX graph fusion.
7. `mx.eval` batching; avoid re-materializing the compute graph each step.
8. Warmup/prec-ompile pass on first launch so the JIT'd Metal kernels are cached.
9. **Then** custom Metal kernels *only for measured hotspots* (per the chosen
   "MLX built-ins first" approach): fused 4-bit grouped GEMM, fused 4D-RoPE, fused
   QK^V attention, RMSNorm. Target iOS 26 / Metal 4 (MSL) — gated by availability.
10. (v2) TeaCache step caching; TAESD live preview (from `mlx-taef`).

**Performance target (to validate, not promise):** < 5 s end-to-end for a 1024² t2i on
iPhone 15 Pro; refine against profiling. Profile with Axiom `xcprof` / Instruments.

---

## Package structure (SwiftPM)

```
AestrixEngine/                         # Swift package
├── Package.swift                      # platforms: .iOS(.v26); deps: ml-explore/mlx-swift (pinned)
├── Sources/AestrixEngine/
│   ├── Engine/
│   │   ├── AestrixEngine.swift        # public actor façade (load/configure/generate/edit)
│   │   ├── ModelContainer.swift       # owns submodels, staged load/free, wired-memory tickets
│   │   └── MemoryBudget.swift         # wired-memory policy + memory-warning handling
│   ├── Models/
│   │   ├── Flux2KleinTransformer.swift
│   │   ├── DoubleStreamBlock.swift / SingleStreamBlock.swift / Embeddings.swift / FinalLayer.swift
│   │   ├── FluxVAE.swift
│   │   ├── Qwen3TextEncoder.swift     # wraps mlx-swift-examples Qwen3 + layer extraction
│   │   └── RoPE4D.swift
│   ├── Pipeline/
│   │   ├── Flux2KleinPipeline.swift   # t2i + edit orchestration
│   │   ├── RectifiedFlowScheduler.swift
│   │   └── Conditioning.swift         # txt_ids/img_ids/ref_ids coordinate builders
│   ├── IO/
│   │   ├── ModelDownloader.swift      # HF streaming download, resumable, size-verified
│   │   ├── WeightLoader.swift         # safetensors + index.json → quantized MLXArrays
│   │   └── Tokenizer.swift            # Qwen3 tokenizer + chat template (port tokenizers.json)
│   ├── Image/
│   │   ├── ImageConversion.swift      # UIImage ↔ MLXArray, normalize [-1,1], /16 crop
│   │   └── VAEPreview.swift           # (v2) TAESD preview hook
│   └── Support/ Errors.swift, Logging.swift, OSVersionGates.swift
├── Sources/AestrixEngineC/ (if any C/MSL bridge for custom kernels — Phase 7)
└── Tests/AestrixEngineTests/          # parity + unit + e2e (see Verification)

AestrixDemo/                           # thin iOS app (Xcode project)
├── GenerationView (prompt → image), EditView (photo + instruction → image),
│   progress / step streaming, memory-pressure UI, settings (steps, size, seed).
```

**Public API (target shape):**
```swift
public actor AestrixEngine {
    public func load(modelDir: URL) async throws -> LoadReport   // staged, reports wired bytes
    public func generate(prompt: String, config: GenConfig) async throws -> UIImage
    public func edit(image: UIImage, instruction: String, config: GenConfig) async throws -> UIImage
    public func unload() async throws                            // free all wired memory
}
// GenConfig: size, steps(=4), seed, guidance(=1.0), dtype
```

---

## Model delivery (download on first launch)

- Stream from `https://huggingface.co/mlx-community/flux2-klein-4b-4bit/resolve/main/<path>`
  per file: `text_encoder/{0,1}.safetensors`, `transformer/{0,1}.safetensors`,
  `vae/0.safetensors`, `tokenizer/*`, the three `*.index.json`.
- `URLSession` background + resumable downloads; verify against known sizes (from the API
  `/api/models/...?blobs=true` — sizes captured in research). Store under
  `Application Support/Aestrix/models/flux2-klein-4b-4bit/`.
- Download order can mirror load order (text_encoder → transformer → vae) so generation
  can start before the full 4.6 GB is on disk (optional v1.1 nicety; v1 = full download).
- HF may rate-limit; document a CDN/mirror override in config. No inference-time network.

---

## Phased roadmap

| # | Milestone | Exit criterion |
|---|---|---|
| 0 | **Scaffold** — SwiftPM package, iOS 26 demo shell, pinned MLX-Swift dep, build green on iPhone 15 Pro sim + device | App launches, imports MLX |
| 1 | **IO** — downloader + safetensors/quantized loader + Qwen3 tokenizer (chat template) parity | Loads all 4.6 GB, tokenizer matches Python token ids |
| 2 | **VAE** — port encoder/decoder; round-trip parity vs Python (encode→decode pixel diff) | UIImage in → UIImage out, <1px drift |
| 3 | **Text encoder** — Qwen3 load, extract layers [9,18,27], concat → (512,7680) parity | ctx matches Python within fp16 tol |
| 4 | **Transformer + RoPE4D** — single `forward()` parity vs Python reference at fixed input | Output matches within fp16 tol |
| 5 | **Pipeline (t2i)** — 4-step loop, scheduler, noise, decode → **first image** | Coherent 1024² image from prompt |
| 6 | **Single-image edit** — ref VAE encode + token concat + t=10 coords | "turn cat into dog" works |
| 7 | **Optimization** — compile, MLXFast, profile, custom Metal-4 kernels for hotspots | Hits perf target; peak wired < budget |
| 8 | **Harden** — memory-warning handling, thermal/backoff, demo polish, error UX | Survives pressure + 50-run soak |

---

## Execution: GitHub repo + M1 (current)

### Step A — publish to GitHub
1. `git init` at repo root `/Users/patricedery/Coding_Projects/Aestrix`; the existing `.gitignore`
   already excludes `.build/`, `DerivedData/`, `*.xcodeproj`, `*.safetensors`.
2. Create the remote via `github.create_repository` (name **Aestrix**, **private**, add an
   **Apache-2.0** `LICENSE`). Copy this plan into the repo as `docs/PLAN.md`.
3. First commit (M0 scaffold + READMEs + plan), `git remote add origin`, `git push -u origin main`.

### Step B — M1: IO layer

**Verified facts (from the model repo):**
- Quantization = 4-bit, **group_size 64** (mflux 0.17.5). Each quantized linear stores
  `.weight` + `.scales` + `.biases` → maps 1:1 onto `MLXNN.QuantizedLinear` (weight / scales /
  biases / groupSize / bits) in `Source/MLXNN/Quantized.swift`.
- Sharding via `<sub>/model.safetensors.index.json` → `weight_map: tensor → shard`. Files:
  `text_encoder/{0,1}.safetensors` (2.26 GB), `transformer/{0,1}.safetensors` (2.18 GB),
  `vae/0.safetensors` (0.17 GB), `tokenizer/*`, 3× `index.json`.
- Tokenizer = `Qwen2Tokenizer` (BPE), `tokenizer/tokenizer.json` (HF fast format); eos
  `<|im_end|>`, pad `<|endoftext|>`; chat template in `tokenizer/chat_template.jinja`.

**MLX-Swift / dep APIs to use:**
- `MLX.loadArrays(url:)` / `loadArraysAndMetadata(url:)` — one safetensors shard → `[String: MLXArray]` (`Source/MLX/IO.swift`).
- **New dependency** `huggingface/swift-transformers` `1.1.0` (product `Transformers`) → `Tokenizers.Tokenizer.from(modelDirectory:)` loads `tokenizer.json`. Add to `AestrixEngine/Package.swift`.
- Quantized module wiring (`QuantizedLinear`, `quantize(model:)`, `Module.update`) lands in M3/M4; **M1 only loads raw arrays + metadata.**

**Files to create (`Sources/AestrixEngine/IO/`):**
- `ModelDownloader.swift` — `actor`; fetches the file list from
  `https://huggingface.co/mlx-community/flux2-klein-4b-4bit/resolve/main/<path>`;
  `URLSession` background config + resumable `URLSessionDownloadTask` (resume data on cancel);
  per-file size verification (expected bytes from `/api/models/...?blobs=true`); progress
  callback; store under `Application Support/Aestrix/models/flux2-klein-4b-4bit/`; skip files
  already present and size-matched.
- `WeightLoader.swift` — `loadWeights(submodel:)`: parse `<sub>/model.safetensors.index.json`
  weight_map → load each shard via `loadArrays`, merge to `[String: MLXArray]`; expose
  quantization metadata (group_size/bits from the index). Return the merged dict.
- `Tokenizer.swift` — wraps `Transformers.Tokenizer`; `encode(prompt:)` applies the Qwen3 chat
  template (load `chat_template.jinja`; fall back to the known string
  `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`) → ids padded/truncated to 512.
- Wire `AestrixEngine.load(modelDir:)` → ensure-downloaded → ready state (weight application
  deferred to M3/M4).

**Tests (`Tests/AestrixEngineTests/`):**
- `TokenizerTests` — encode a fixed prompt; assert ids equal a Python `transformers`
  Qwen2Tokenizer reference (generated once with `tokenizers`); assert length 512.
- `WeightLoaderTests` — load `vae/0.safetensors`; assert expected tensor names/shapes present
  and quant metadata `group_size == 64`.
- `DownloaderTests` — download one small file (`tokenizer_config.json`), verify size, skip-if-present.

**M1 verification:** `swift test` for MLX-free parts; `TEST_RUNNER_AESTRIX_RUN_MLX_TESTS=1
xcodebuild test -scheme AestrixEngine -destination 'platform=macOS'` for the MLX weight-load
test; tokenizer parity vs Python. (Full 4.6 GB download verified manually/on-device; unit
tests use the small VAE + tokenizer.)

---

## Key risks & mitigations

| Risk | Mitigation |
|---|---|
| **iOS wired-memory ceiling lower than macOS** (may reject ~4 GB wired) | Validate Phase B on iPhone 15 Pro in Milestone 4; fallbacks: 768², mixed precision, more aggressive free |
| **MLX "research not production" / API churn** | Pin MLX-Swift version in `Package.swift`; isolate MLX touches behind `ModelContainer` |
| **4-bit dequant-GEMM is the hot path** | Measure first; if slow, write fused grouped-GEMM Metal kernel (Phase 7) |
| **Qwen3 loader complexity** | Reuse mlx-swift-examples Qwen3 (quantized) rather than hand-rolling; only add layer-extraction hook |
| **Quant format mismatch** (group_size/bits) | Loader reads `index.json` quantization config; assert against MLXNN quantized-Linear expectation |
| **Attention memory over 4600 tokens** | `mlx.fast.scaled_dot_product_attention` (flash); tile if needed |
| **First-run JIT compile latency** | Warmup/prec-compile pass on load; cache metallib |
| **Thermal throttling** | Design for bursts; expose progress; optional step-count reduction under thermal pressure |
| **App Store / privacy / safety** | On-device = no inference network (privacy win). Model is Apache-2.0. NSFW/provenance filters are host-app policy — engine exposes hooks, leaves policy to the app |

---

## Verification

- **Parity (correctness gate, each component):** fixed seed + fixed input → Swift output
  vs Python reference (`mflux`/`mlx-examples`) within fp16 tolerance. Covers tokenizer,
  VAE, text encoder, transformer forward, full pipeline.
- **End-to-end image:** generate a reference prompt at seed 0; compare to a Python-baseline
  image (pixel/CLIP-similarity). Editing: before/after on a known image.
- **Memory:** Instruments/`xctrace` peak wired bytes < budget; confirm Qwen3 freed before
  denoise; survives `didReceiveMemoryWarning`; 50-run soak no growth.
- **Performance:** per-step + total time logged; meet target on iPhone 15 Pro.
- **Tooling:** Axiom `xcprof`/performance-profiler agent for on-device CPU/GPU traces;
  `session_show_defaults` → `build_run_sim` / device build via XcodeBuildMCP.

---

## Reference sources (porting & architecture)

- Model: [mlx-community/flux2-klein-4b-4bit](https://hf.co/mlx-community/flux2-klein-4b-4bit)
  · base [black-forest-labs/FLUX.2-klein-4B](https://hf.co/black-forest-labs/FLUX.2-klein-4B)
- Pipeline mechanics: [FLUX.2 Klein — How Inference Works (Geronimo)](https://medium.com/@geronimo7/flux-2-klein-how-inference-works-05553fcdbe7e)
- Python MLX references: [filipstrand/mflux](https://github.com/filipstrand/mflux) ·
  [ml-explore/mlx-examples/flux](https://github.com/ml-explore/mlx-examples/tree/main/flux) ·
  [IonDen/mlx-taef](https://github.com/IonDen/mlx-taef) · [IonDen/mlx-teacache](https://github.com/IonDen/mlx-teacache)
- MLX-Swift: [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift) (iOS `.v17+`,
  wired-memory tickets, `MLXFast`, `compile`)
- Apple guidance: WWDC25 Session 298 *Explore LLMs on Apple Silicon with MLX* ·
  Swift.org *On-Device ML Research with MLX and Swift*
