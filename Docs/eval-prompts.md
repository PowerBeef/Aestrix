# Eval prompts (seeded from BFL flux-best-practices)

Use for smoke tests and visual quality gates. Klein prefers **narrative prose**, **front-loaded subject**, **strong lighting**, moderate length (~40–70 words). **No negative prompts.**

**After generating any sample, run the full eval procedure:**  
[`Docs/EVAL_WORKFLOW.md`](EVAL_WORKFLOW.md) (`--analyze --vision-brief`, then vision checklist).

## T2I (distilled 4-step)

1. **Lighting / atmosphere (BFL klein style)**  
   A cozy coffee shop interior bathed in warm afternoon light, steam rising lazily from ceramic cups, worn leather armchairs arranged around small wooden tables, bookshelves lining exposed brick walls, dust motes floating in sunbeams through tall windows.

2. **Subject first + rim light**  
   A weathered fisherman in his 70s with deep wrinkles and a salt-and-pepper beard, wearing a navy cable-knit sweater, standing at the helm of his wooden boat. Golden hour sunlight from the left creates dramatic rim lighting on his profile. Shallow depth of field with harbor lights soft in the background.

3. **Hex color**  
   A modern product photo of a matte ceramic mug in #C45C26 terracotta orange on a #F5F0E8 linen surface, soft window light from the left, subtle shadow, clean commercial still life.

4. **Typography smoke**  
   A minimal poster with the headline "OPEN STUDIO" in bold condensed white sans-serif on a deep indigo #1A1A40 background, soft gradient, centered composition, print-ready graphic design.

5. **Short scramble check** (must remain coherent despite short prompt)  
   A red fox in a snowy forest at sunrise, photorealistic.

## I2I (strength path)

Prompt style: state **what changes** and **what stays**. Curves and bands: [`I2I_STRENGTH.md`](I2I_STRENGTH.md).

1. Change the background to a tropical beach at sunset while keeping the subject’s pose, clothing, and expression exactly the same.  
2. Transform to golden hour lighting with warm tones and long shadows, maintaining the exact composition and photorealistic style.  
3. Convert this photograph to a detailed pencil sketch with careful shading, keeping composition identical.

## I2I identity (Tier B — `--identity`)

Use for people/character consistency. Prefer **strength ≥ 0.85–0.9**. Be explicit about face/pose vs wardrobe/scene.

1. **Recolor only:** Same person, exact face and pose. Keep the exact same silk blouse cut, collar, and draping; change only the fabric color to deep emerald green #0B5F4B. Warm golden-hour window light.  
2. **Wardrobe + light:** Same woman, identical face shape, eyes, brows, nose, lips, freckles, hairstyle, head angle and camera framing. Outdoor Mediterranean balcony at golden hour, warm side light from the left, deep emerald green silk top. Photoreal portrait, neutral expression.  
3. **New outfit (explicit):** Same person, exact face and pose. Replace the ivory blouse with a different deep emerald green silk wrap top (new neckline and cut). Golden hour outdoor balcony.

## Seeds

Use fixed seeds (`42`, `0`, `7`) for regression comparisons across quant tiers.
