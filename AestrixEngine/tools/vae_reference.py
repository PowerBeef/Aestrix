#!/usr/bin/env python3
"""Generate a Python (mflux/MLX) reference for the FLUX.2 VAE encode/decode.

The VAE mid-block attention linears are stored 4-bit; mflux's Flux2VAE uses plain
nn.Linear, so we dequantize them back to bf16 with mx.dequantize before load_weights.

Outputs a safetensors file with `input`, `latent`, `decoded` in NHWC (the layout
AestrixEngine's FluxVAE uses) so Swift parity tests compare directly.

Run from AestrixEngine/:
    .build/venv/bin/python tools/vae_reference.py [<model_dir>] [<out.safetensors>]
"""
import os
import sys

import numpy as np
import mlx.core as mx
from mlx import nn
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE

# --fp32 runs the VAE in float32 (mflux default is bf16). Used to validate the Swift port
# independent of MLX-Metal bf16 non-determinism (float32 reductions are deterministic).
fp32 = "--fp32" in sys.argv
argv = [a for a in sys.argv[1:] if not a.startswith("--")]
model_dir = argv[0] if len(argv) > 0 else ".build/fixtures/flux2-klein-4b-4bit"
suffix = "_fp32" if fp32 else ""
if fp32:
    ModelConfig.precision = mx.float32
out_path = argv[1] if len(argv) > 1 else f"{model_dir}/parity/vae_reference{suffix}.safetensors"
print(f"precision = {ModelConfig.precision}")

raw = mx.load(f"{model_dir}/vae/0.safetensors")

# Dequantize 4-bit attention linears (those carrying a companion `.scales` tensor).
quantized = {k[:-7] for k in raw if k.endswith(".weight") and (k[:-7] + ".scales") in raw}
weights = {}
for k, v in raw.items():
    if k.startswith("__"):
        continue
    base = k[:-7] if k.endswith(".weight") else None
    if base in quantized and k.endswith(".weight"):
        weights[k] = mx.dequantize(v, raw[base + ".scales"], raw[base + ".biases"], group_size=64, bits=4)
    elif (k.endswith(".scales") or k.endswith(".biases")) and k[:-7] in quantized:
        continue  # folded into the dequantized weight
    else:
        weights[k] = v

vae = Flux2VAE()
vae.load_weights(list(weights.items()))

# Deterministic NCHW input in [-1, 1] (mflux's VAE public layout).
np.random.seed(0)
img = mx.array(np.random.uniform(-1.0, 1.0, (1, 3, 128, 128)).astype(np.float32))

# Capture encoder stages (mflux encoder works in NCHW internally) to localize parity drift.
to_nhwc = lambda a: mx.transpose(a, (0, 2, 3, 1))
stages = {}
h = vae.encoder.conv_in(img);                       stages["conv_in_out"] = to_nhwc(h)
for db in vae.encoder.down_blocks: h = db(h);       stages["after_down_blocks"] = to_nhwc(h)
h = vae.encoder.mid_block(h);                       stages["after_mid_block"] = to_nhwc(h)
h = vae.encoder.conv_norm_out(h)
h = nn.silu(h).astype(mx.bfloat16)
enc = vae.encoder.conv_out(h);                      stages["encoder_out"] = to_nhwc(enc)  # moments (64-ch)

latent = vae.encode(img)          # [1, 32, 16, 16]  (mode; scale=1, shift=0)

# Decoder stages (NCHW internally): post_quant_conv → conv_in → mid → up_blocks → norm → silu → conv_out
dlat = mx.transpose(latent, (0, 2, 3, 1))                 # NCHW → NHWC
dlat = vae.post_quant_conv(dlat)
dlat = mx.transpose(dlat, (0, 3, 1, 2))                   # → NCHW for decoder
h = vae.decoder.conv_in(dlat);                             stages["dec_conv_in"] = to_nhwc(h)
h = vae.decoder.mid_block.resnets[0](h);                   stages["dec_mid_res0"] = to_nhwc(h)
h = vae.decoder.mid_block.attentions[0](h);                stages["dec_mid_attn"] = to_nhwc(h)
h = vae.decoder.mid_block.resnets[1](h);                   stages["dec_mid"] = to_nhwc(h)
for i, ub in enumerate(vae.decoder.up_blocks):
    for r in ub.resnets: h = r(h)
    stages[f"dec_up{i}_res"] = to_nhwc(h)                  # after resnets, before upsample
    for up in ub.upsamplers: h = up(h)
    stages[f"dec_up{i}"] = to_nhwc(h)
stages["dec_after_up"] = stages[f"dec_up{len(vae.decoder.up_blocks)-1}"]
h = vae.decoder.conv_norm_out(h)
h = nn.silu(h).astype(mx.bfloat16)
dec = vae.decoder.conv_out(h);                             stages["dec_out"] = to_nhwc(dec)

decoded = vae.decode(latent)      # [1, 3, 128, 128]

os.makedirs(os.path.dirname(out_path), exist_ok=True)
mx.save_safetensors(out_path, {
    "input": to_nhwc(img),
    **stages,
    "latent": to_nhwc(latent),
    "decoded": to_nhwc(decoded),
})
print(f"stages {list(stages)}  latent {latent.shape}  decoded {decoded.shape}  -> {out_path}")
