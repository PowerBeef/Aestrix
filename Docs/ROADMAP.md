# Aestrix roadmap

**Last updated:** 2026-08-11  
**Working tree focus:** macOS library + CLI is the shipping surface for now.  
**Remaining work** is parked here so agents and humans can resume without rediscovering context.

---

## Done (v1 macOS core)

| ID | Item | Where / notes |
|----|------|----------------|
| P0 | Scaffold, SPM packages, staged memory policy, BFL skills | `Package.swift`, `Docs/MEMORY.md`, `.grok/skills/flux-best-practices/` |
| P1 | Pure math: RoPE, temb, modulation, flow-match scheduler | `AestrixCore` |
| P2 | MMDiT + 4-bit load | `AestrixDiT`, `aestrix load-dit` |
| P3 | VAE encode/decode + pack/BN path | `AestrixVAE`, `aestrix load-vae` |
| P4 | Qwen3 TE, chat template, BPE, 9/18/27 tap | `AestrixText`, `aestrix load-te` / `encode-prompt` |
| P5 | Staged T2I | `AestrixPipeline.generate`, `aestrix t2i` |
| P6 | Strength I2I (full N-step strength schedule) | `AestrixPipeline.edit`, `aestrix i2i` |
| P6b | Pixel + vision eval workflow | `AestrixEval`, `Docs/EVAL_WORKFLOW.md`, `--analyze` |

**Resume baseline (macOS):**

```bash
swift build && ./Scripts/ensure-metallib.sh
# Snapshot: ~/Library/Caches/Aestrix/models/mlx-community--FLUX.2-Klein-4B-4bit
.build/debug/aestrix t2i "…" --output /tmp/out.png --analyze --vision-brief
```

---

## Parked — resume later

Status legend: `parked` = not started · `partial` = some code/docs · `blocked` = needs decision

### P7 — iOS host (next major phase)

| Field | Detail |
|-------|--------|
| **Status** | `parked` |
| **Goal** | Ship an iOS 26 app (or host) that links `AestrixRuntime` and runs staged T2I (+ I2I) on-device |
| **Depends on** | macOS path stable (done); full metallib packaging for app targets; device memory tiers |

**Acceptance criteria**

- [ ] Xcode app target (or multiplatform package consumer) links `AestrixRuntime` / MLX  
- [ ] Metallib / MLX resources packaged for iOS (not only SPM CLI bootstrap)  
- [ ] On-device snapshot path (app container / shared group) documented  
- [ ] Single 512² T2I smoke on a physical device or iOS simulator (if Metal allows)  
- [ ] Tier-aware max side / memory policy (`DeviceTier`, `MemoryPolicy.staged`)  
- [ ] Basic UI: prompt, generate, progress, save/share  
- [ ] Optional: I2I with photo picker + strength slider  
- [ ] Eval notes for device outputs (pixel harness runs on host Mac from exported PNG if needed)

**Resume notes**

- Reuse `AestrixPipeline` actor; do not reimplement TE/DiT/VAE in the app.  
- Product locks still apply: Klein 4B only, prequant only, staged default.  
- Start with Tier L (512², 4-step) before higher resolutions.  
- See `Docs/MEMORY.md` for peak budgets; `Docs/WEIGHTS.md` for pack layout.

**Suggested first PR**

1. Empty iOS app shell + SPM dependency on local `Aestrix` packages  
2. Wire metallib / MLX init parity with `MLXBootstrap`  
3. Call `AestrixPipeline.generate` from a button with hardcoded prompt  

---

### P8 — macOS polish / regression (optional before or after iOS)

| Field | Detail |
|-------|--------|
| **Status** | `parked` |
| **Goal** | Harden macOS CLI for regressions and measured memory |

**Backlog**

- [ ] Scripted regression from `Docs/eval-prompts.md` with fixed seeds (42, 0, 7)  
- [ ] `aestrix bench-mem` — measured RSS around TE / DiT / VAE stages  
- [ ] Pin Hub `revision` SHA in config / docs when shipping a release  
- [ ] Golden image or metric floors in CI (pixel-only; vision stays agent/human)  
- [ ] Document known I2I strength curves for color vs structure edits  

---

### Out of v1 scope (track only — do not start without product decision)

| Item | Why deferred | Notes if resumed |
|------|--------------|------------------|
| Klein 9B / FLUX.2 Dev | Non-Commercial / size | Different product locks |
| Multi-reference edit | Scope | Diffusers-style token concat; higher peak RAM |
| CFG / negative prompts | Distilled klein defaults | Not used on 4-step path |
| LoRA training / load | Scope | Separate product surface |
| User-facing bf16 | Product lock | Maintainer-only tools only |
| Prompt upsampling | Product lock | Optional later UX |
| VLM-in-CI vision review | Infra cost | Pixel gate first |

---

## How to resume (agents)

1. Read this file + `AGENTS.md` product locks.  
2. Pick the highest-priority `parked` ID (default **P7**).  
3. Confirm macOS smoke still works (`t2i` + `EVAL_WORKFLOW.md`).  
4. Open a focused branch; do not expand into “Out of v1” without an explicit ask.  
5. Update this roadmap (checkboxes + “Last updated”) when an item lands or is cancelled.

---

## Decision log (short)

| Date | Decision |
|------|----------|
| 2026-08 | Pre-quant only; default 4-bit; no user bf16 |
| 2026-08 | Staged TE→DiT→VAE is default memory policy |
| 2026-08 | Klein 4B only (Apache-2.0) |
| 2026-08 | I2I uses full N-step strength schedule (not truncated T2I slice) |
| 2026-08 | Gen quality = pixel harness + vision checklist (`EVAL_WORKFLOW.md`) |
| 2026-08 | Park remaining work; macOS CLI is current focus surface |
