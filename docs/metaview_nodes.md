# MetaView nodes (novel-view synthesis)

MetaView ports the [MetaView / Depth-Anything-3] novel-view-synthesis pipeline to
ComfyUI. Given one input image and a target camera (yaw / pitch / radius), it
renders the scene from the new viewpoint using a grafted Qwen-Image-Edit DiT.

Three nodes, all under the **EnviralDesign/MetaView** category. See
`docs/metaview_contract.md` for the exact tensor/coordinate contract between the
conditioning node and the model.

---

## Model file locations

The nodes create **no** junctions, hardlinks, or symlinks. Place (or point your
model manager at) the weights in the standard ComfyUI model folders:

| What | Where | Notes |
|------|-------|-------|
| MetaView DiT artifact | `models/diffusion_models/metaview_dit_fp8scaled.safetensors` | fp8-scaled; shows in the `MetaViewModelLoader` dropdown. |
| DA3 3D-feature model | `models/depthanything3/` — `da3_giant.safetensors` (single file) or `DA3-GIANT-1.1/` (HF dir) | [depth-anything/DA3-GIANT-1.1](https://huggingface.co/depth-anything/DA3-GIANT-1.1) |
| DA3 depth model | `models/depthanything3/` — `da3nested_giant_large.safetensors` or `DA3NESTED-GIANT-LARGE-1.1/` | [depth-anything/DA3NESTED-GIANT-LARGE-1.1](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1) |

Single-file checkpoints follow the community DA3 naming (`da3_giant`, `da3nested_giant_large`, ...);
the node maps the filename to the bundled model preset. HF-style directories
(`config.json` + `model.safetensors`) work too. Legacy `models/depth_anything_3/` is
still scanned for back-compat.
| Qwen2.5-VL text encoder | `models/text_encoders/.../qwen_2.5_vl_7b_fp8_scaled.safetensors` | loaded via stock `CLIPLoader`, type `qwen_image`. |
| Qwen-Image VAE | `models/vae/.../qwen_image_vae.safetensors` | loaded via stock `VAELoader`. |

`models/depthanything3` is registered as a ComfyUI model-folder key on first
use. If it is empty the DA3 dropdowns are simply empty (no crash) — drop the DA3
model directories in and refresh.

---

## MetaViewModelLoader

Loads the grafted MetaView DiT onto ComfyUI's native fp8 quantized op path and
returns a standard `MODEL`.

- **Inputs (required)**
  - `model_name` (dropdown over `models/diffusion_models`) — the MetaView DiT
    safetensors (e.g. `metaview_dit_fp8scaled.safetensors`).
- **Outputs**
  - `model` (`MODEL`) — patchered MetaView model. Registers a Qwen-Image /
    flow-matching sampling config with `latent_format = Wan21`.

The loader synthesizes `comfy_quant` markers for the fp8 layers with
`full_precision_matrix_mult: true` (Comfy-Org scaled-fp8 convention): weights
stay fp8 on disk/VRAM and are dequantized to bf16 for matmul — matching the
MetaView reference `--no_fp8_mm` path.

## MetaViewDA3Loader

Loads the Depth-Anything-3 GIANT (3D-feature) model and the nested depth model
into a lazily-loaded `DA3_MODEL` bundle. The models load on first use and offload
to CPU after each pass so the DiT can claim VRAM.

- **Inputs (required)**
  - `giant_model` (dropdown over `models/depthanything3`) — DA3 dir providing
    the 3D feature volume + predicted intrinsics.
  - `depth_model` (dropdown over `models/depthanything3`) — DA3 nested dir
    providing the depth map.
- **Outputs**
  - `da3_model` (`DA3_MODEL`) — the bundle consumed by `MetaView3DConditioning`.

## MetaView3DConditioning

Runs DA3 on the input image, derives the target camera from yaw/pitch/radius,
and augments the Qwen-Image-Edit conditioning with the MetaView geometry keys
plus a correctly-sized sampler `LATENT`.

- **Inputs (required)**
  - `conditioning` (`CONDITIONING`) — from `TextEncodeQwenImageEdit`
    (the trigger prompt encoded with the source image; see wiring note below).
  - `da3_model` (`DA3_MODEL`) — from `MetaViewDA3Loader`.
  - `image` (`IMAGE`) — the source view.
  - `yaw` (`FLOAT`, deg) — target-camera yaw. Negative = orbit one way, positive
    the other.
  - `pitch` (`FLOAT`, deg) — target-camera pitch.
  - `radius` (`FLOAT`) — orbit radius; `0` or `auto_radius` derives it from the
    centre-pixel depth.
  - `auto_radius` (`BOOLEAN`) — derive radius from centre depth (recommended).
- **Inputs (optional)**
  - `vae` (`VAE`) — encodes the (gen-dim-resized) source image into the
    reference latent the DiT consumes as the source-camera tokens. **Supply this
    here** (see wiring note) rather than on the text-encode node.
  - `width` / `height` (`INT`) — `0` = auto (aspect-preserving, area-matched to
    960x528, multiple of 16).
- **Outputs**
  - `conditioning` (`CONDITIONING`) — positive conditioning carrying
    `metaview_viewmats/ks/feat3d/depth` + the reference latent.
  - `latent` (`LATENT`) — empty latent at the generation resolution; feed to
    `SamplerCustom.latent_image` and `MetaViewSigmas.latent`.

## MetaViewSigmas

Emits the exact Qwen-Image / MetaView FlowMatch sigma schedule (exponential mu
from patch-grid length + terminal shift to 0.02). Pack-local — does **not**
register into Comfy's global scheduler table.

- **Inputs**
  - `steps` (`INT`, default 8)
  - `latent` (optional `LATENT`) — preferred; mu from latent spatial size
  - `width` / `height` — fallbacks when no latent is wired
- **Outputs**
  - `sigmas` (`SIGMAS`) — wire to stock `SamplerCustom` / `SamplerCustomAdvanced`

---

## Sample graph (see `workflows/metaview_example_api.json`)

```
LoadImage ─┬─────────────────────────────► MetaView3DConditioning.image
           │                                        ▲   │
CLIPLoader ─┴─► TextEncodeQwenImageEdit ─────────────┘   │ (cond)   (latent)
(qwen_image)     (prompt=镜头视角转到指定位置,           │      │        │
                  image=LoadImage, NO vae)               │      │        │
VAELoader ──────────────────────────────► MetaView3DConditioning.vae     │
   │                                                     │               │
   │                          MetaViewDA3Loader ─► MetaView3DConditioning.da3_model
   │                                                     │ (cond)        │ (latent)
MetaViewModelLoader ─┐                                   ▼               ▼
TextEncodeQwenImageEdit("") ─┐              positive            latent ──┬─► MetaViewSigmas
                             │                                           │         │
                    SamplerCustom ◄── KSamplerSelect(euler) ◄────────────┘         │
                    (cfg 1.0, seed 42, sigmas from MetaViewSigmas) ◄────────────────┘
                                          │
                                          ▼
                                       VAEDecode ─► SaveImage (prefix metaview_e2e)
```

### Wiring notes (important)

1. **Reference latent goes through the conditioning node, not the text encoder.**
   `TextEncodeQwenImageEdit` receives `clip`, `prompt`, and `image`, but **not**
   `vae`. With a `vae` it would attach its own ~1MP `reference_latents[0]` at the
   wrong resolution. `MetaView3DConditioning` (given the `vae`) produces the
   single reference latent at the generation resolution, which the DiT consumes
   as `reference_latents[0]` (the source-camera tokens). Passing `image` (without
   `vae`) still lets the VL encoder attend to the source image for the trigger
   phrase.

2. **Trigger prompt** is fixed: `镜头视角转到指定位置` ("move the camera to the
   specified viewpoint"). The model does not re-encode text; this is the
   convention it was trained on.

3. **Negative conditioning** is a second `TextEncodeQwenImageEdit` with an empty
   prompt (clip only). At **CFG 1.0** ComfyUI skips the unconditional pass, so
   the negative is never actually evaluated — any valid empty CONDITIONING works
   (`ConditioningZeroOut` on the positive is an equivalent alternative).

4. **Sampler settings** are fixed by the fused Lightning weights: **8 steps,
   CFG 1.0, euler, simple scheduler**. The `latent_image` must come from
   `MetaView3DConditioning`'s `latent` output (correct gen resolution).
