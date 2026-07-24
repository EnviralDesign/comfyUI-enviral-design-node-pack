# MetaView DiT — ComfyUI integration contract

Status: authoritative. Owned by the model-port agent (`metaview_model.py`).
Consumers: the conditioning agent (`metaview_conditioning.py`, `metaview_da3/`) and the
workflow/orchestrator.

This document defines exactly what the MetaView DiT consumes per denoise step, the tensor
shapes / dtypes / coordinate conventions, and how the conditioning node must hand data to the
model. It is derived from `C:\repos\MetaView\src\MetaView_pipeline.py`
(`model_fn_MetaView`, ~line 436+), `C:\repos\MetaView\src\PRoPE.py`, and
`C:\repos\MetaView\src\MetaView_dit.py`.

---

## 1. Model summary

MetaView DiT = Qwen-Image-Edit transformer (60 double-stream blocks, dim 3072, 24 heads,
head_dim 128) + an additive **novel-view-synthesis graft**:

- `_3D_in`: `Linear(6144 -> 3072)` — projects the DA3 3D feature into the transformer width.
- Per block `prope_attn` (`MetaViewSelfAttention3D`): a parallel self-attention branch that
  jointly attends the image tokens and the projected 3D-feature tokens, both positionally
  encoded with **PRoPE** (Cameras as Relative Positional Encoding). Its output is added to the
  main image attention output. Zero-initialised output projection at train start, so the graft
  is additive.
- **PRoPE** runs on the **parallel** `prope_attn` branch (image + 3D-feature tokens).
  The main Qwen double-stream attention keeps standard Qwen 2D RoPE on image tokens
  (`prope=None` when `add_attn=True`). Text tokens always use Qwen RoPE.

Runtime config (from the executing demo path `src/inference_lowvram.py`, not pipeline
signature defaults): `add_attn_type='self_attn_3D'`, `merge_3D=True`, `_3d_dim=6144`,
`add_in_dim=3072`, `prope_dim_arrange=[64, 20, 20, 24]` (4 terms → depth channel active),
`decode_3D=False`, `add_attn=True`.

Sampling: **8 steps, CFG scale 1.0** (Qwen-Image-Edit-Lightning-8steps-V1.0 fused into the
weights). Use the pack's `MetaViewSigmas` + stock `SamplerCustom` for the Qwen-Image
FlowMatch schedule (terminal shift to 0.02). Because CFG == 1.0, ComfyUI skips the
unconditional pass — MetaView conditioning only needs to live on the **positive** conditioning.

---

## 2. Geometry / grid rule (MUST match on both sides)

Let the input image have aspect ratio `r = W_in / H_in`.

1. **Generation size**: area-match to `960 x 528` (`AREA = 506880`), preserving aspect,
   each side a multiple of **16**:
   ```
   gen_w = round(sqrt(AREA * r) / 16) * 16
   gen_h = round(sqrt(AREA / r) / 16) * 16
   ```
   (Reference target is 960x528 landscape; portrait inputs swap.)
2. **Latent size** (Qwen/Wan2.1 VAE, 16 channels, /8): `H_lat = gen_h/8`, `W_lat = gen_w/8`.
3. **Patch grid** (patchify P=Q=2 → /16 from pixels): `patches_y = gen_h/16`,
   `patches_x = gen_w/16`. For 960x528 → `patches_x = 60`, `patches_y = 33`.
4. **DA3 processing resolution**: `process_res = max(gen_w, gen_h) * 14 // 16`.
   DA3 output 3D feature is resized to the patch grid `(patches_y, patches_x)` before it is
   handed to the model (see feat_3D below).

The model derives `patches_x/patches_y` from the latent it is given; the conditioning node must
produce `feat_3D`, `viewmats`, `Ks` consistent with the **same** `gen_w/gen_h`.

---

## 3. Conditioning keys (what the conditioning node sets)

Set these on the **positive** CONDITIONING via
`node_helpers.conditioning_set_values(cond, {...})`. The model reads them in
`BaseModel.extra_conds` and bundles them (see §5). All are optional individually, but PRoPE only
activates when `metaview_viewmats` **and** `metaview_ks` are both present; the 3D graft only
activates when `metaview_feat3d` is also present.

| key                | dtype    | shape                         | notes |
|--------------------|----------|-------------------------------|-------|
| `metaview_viewmats`| float32  | `[2, 4, 4]` or `[b, 2, 4, 4]` | camera extrinsics (world→cam), order (target, source), see §4 |
| `metaview_ks`      | float32  | `[2, 3, 3]` or `[b, 2, 3, 3]` | intrinsics in the **PRoPE-native** form, fed as-is; see §4 |
| `metaview_feat3d`  | bf16     | `[Hp, Wp, 6144]` or `[b, Hp, Wp, 6144]` | DA3 3D feature at patch grid, see §4 |
| `metaview_depth`   | float32  | `[2, H, W]` or `[b, 2, H, W]` | raw depth (not VAE-encoded); **required for shipped PRoPE** (see §4) |

A missing batch dim is auto-promoted to `b=1` by the model. The model bundles these in a
`CONDConstant` (a plain dict) so ComfyUI does not silently cast the float32 camera matrices to
bf16; it then upcasts `viewmats`/`Ks` to float32 before PRoPE (which itself upcasts internally).
Passing float32 is the faithful choice — the reference pipeline happens to cast the camera
matrices to bf16 before PRoPE re-upcasts them, so float32 conditioning is a strict superset in
precision.

### Reference (edit) latent — the input-image ride-along

The source/input image is provided the **stock Qwen-Image-Edit way**: VAE-encode it and attach
it as `reference_latents` (this is exactly what the stock `TextEncodeQwenImageEdit` node does).
The model consumes `reference_latents[0]` as the **camera-1 (source)** image tokens, patchified
and concatenated to the target tokens MetaView-style (plain token concat at the same patch grid;
**not** the comfy "kontext" offset scheme). The reference latent must be the same latent
resolution as the target (`H_lat x W_lat`).

### Trigger prompt

Text is encoded by the stock `TextEncodeQwenImageEdit` (Qwen2.5-VL). Use the project's fixed
trigger-prompt convention for novel-view synthesis; the model does not re-encode text.

---

## 4. Semantics / coordinate conventions

### `metaview_viewmats` — `[n_cams=2, 4, 4]` float32
- SE(3) **world-to-camera** transforms (`camera <- world`), i.e. `X_cam = viewmat @ X_world`
  (homogeneous). This is what PRoPE calls `viewmats` and treats as `camera<-world`.
- **Camera ordering is `(target, source)`**:
  - index 0 = **target** = the novel view being generated.
  - index 1 = **source/edit** = the input reference image's view.
- The graft's `add_PRoPE` (the 3D-feature branch) uses **only camera index 1** (source):
  `viewmats[:, 1:2]`, `Ks[:, 1:2]`.

### `metaview_ks` — `[n_cams=2, 3, 3]` float32
- Intrinsics in the exact form PRoPE consumes them (PRoPE uses `Ks_norm = Ks`, i.e. **no
  re-normalization inside the model** — whatever is supplied is fed through verbatim). The
  reference pipeline (`src/inference.py`) builds this from the DA3-predicted intrinsics
  `intri` as:
  ```
  width  = intri[0,2] * 2      # = 2 * cx
  height = intri[1,2] * 2      # = 2 * cy
  Ks = [[ intri[0,0] / width , 0.0                 , 0.0 ],   # fx / (2*cx)
        [ 0.0                , intri[1,1] / height , 0.0 ],   # fy / (2*cy)
        [ 0.0                , 0.0                 , 1.0 ]]
  ```
  i.e. **focal lengths divided by `2·cx` / `2·cy`, and the principal-point entries set to
  `0.0`** (NOT ~0.5). The same 3×3 is used for both cameras. PRoPE then forms
  `P = lift(K) @ viewmat` (`image<-world`) and `P_inv = inv(viewmat) @ lift(inv(K))`; no skew.
- Same `(target, source)` ordering as viewmats (both cameras carry the identical Ks in the
  reference pipeline).
- The model performs **no normalization or reinterpretation** of `metaview_ks`; it is the
  conditioning node's responsibility to emit this exact convention.

### `metaview_feat3d` — `[Hp, Wp, 6144]`
- DA3 (Depth-Anything-3) **3D feature** for the SOURCE view, laid out on the patch grid:
  `Hp = patches_y = gen_h/16`, `Wp = patches_x = gen_w/16`. Channel dim `6144` (`_3d_dim`).
- The model flattens `[Hp, Wp, 6144] -> [Hp*Wp, 6144]`, projects with `_3D_in` to
  `[Hp*Wp, 3072]`, and treats the `Hp*Wp` tokens as a **single-camera** (source) token grid.
  This is why `Hp*Wp` MUST equal `patches_x * patches_y` — PRoPE's `add_PRoPE` asserts
  `seqlen == 1 * patches_x * patches_y`.

### `metaview_depth` — `[n_cams, H, W]` float32 (raw depth)
- Raw metric depth (NOT VAE-encoded). In the reference demo `depth[:,0]` (target) is all
  zeros and `depth[:,1]` (source) is the DA3 depth of the input, interpolated to `(gen_h, gen_w)`.
- The shipped demo uses `prope_dim_arrange=[64, 20, 20, 24]` (four terms summing to head_dim
  128). `len(dim_arrange)==4` activates the PRoPE depth/z term, so depth **does** affect DiT
  math. The model bilinearly resizes depth to the patch grid before `_precompute_and_cache_apply_fns`.
  (`MetaViewPipeline.__call__`'s signature default `[16,56,56]` is overridden by the demo and
  is not what the released weights expect.)

---

## 5. How the model consumes the keys (informative)

`metaview_model.py` implements a `model_base.BaseModel` subclass whose `extra_conds` reads the
four keys above and bundles them into a single `comfy.conds.CONDConstant` (a plain dict, no
`.dtype`) so ComfyUI does not silently cast the float32 camera matrices to bf16. The grafted
`_forward`:

1. Patchifies the target latent → camera-0 image tokens.
2. Patchifies `reference_latents[0]` → camera-1 (source) image tokens; concatenates → image seq
   of length `2 * patches_x * patches_y`.
3. Builds `PRoPE` (2 cameras + depth) for the parallel 3D-attn branch and `add_PRoPE`
   (source camera only) for the 3D-feature stream, from `viewmats`/`Ks`/`depth`.
4. Runs 60 grafted blocks: standard Qwen double-stream attention (image tokens keep 2D
   Qwen RoPE when `add_attn=True`; text gets Qwen RoPE) **plus** additive `prope_attn`
   (PRoPE on image+3D tokens), added to the image attention output.
5. Unpatchifies camera-0 tokens back to the latent (camera-1/source tokens are discarded).

PRoPE token ordering requirement: the image sequence must be reshapeable to
`(cameras, patches_x, patches_y)` in row-major order — camera 0 (target) tokens first, then
camera 1 (source), each an `Hp x Wp` grid flattened row-major. The model guarantees this by
construction.

---

## 6. Quick checklist for the conditioning node

- [ ] Compute `gen_w/gen_h` from input aspect via the §2 area-match rule.
- [ ] VAE-encode source image → reference latent at `H_lat x W_lat`; attach as
      `reference_latents` (stock TextEncodeQwenImageEdit already does this).
- [ ] Build `viewmats [2,4,4]` float32 (world→cam), order `(target, source)`.
- [ ] Build `Ks [2,3,3]` float32 as `[[fx/(2·cx),0,0],[0,fy/(2·cy),0],[0,0,1]]` (principal
      point entries = 0), identical for both cameras. Fed to PRoPE verbatim — do NOT normalize
      by gen_w/gen_h and do NOT put 0.5 in the principal-point slots.
- [ ] Run DA3 at `process_res = max(gen_w,gen_h)*14//16`; resize its 3D feature to
      `[patches_y, patches_x, 6144]` → `metaview_feat3d`.
- [ ] Run DA3 depth; build `depth [2,gen_h,gen_w]` (ch0 zeros, ch1 source depth) → `metaview_depth`.
- [ ] `conditioning_set_values(positive, {"metaview_viewmats":..., "metaview_ks":...,
      "metaview_feat3d":..., "metaview_depth":...})`.
- [ ] Sample at 8 steps, CFG 1.0, with the MetaView MODEL from `MetaViewModelLoader`.
