"""MetaView (novel-view synthesis) conditioning nodes.

Ports the conditioning side of the MetaView pipeline
(C:/repos/MetaView/src/inference_lowvram.py + src/MetaView_pipeline.py) to
ComfyUI. Two nodes:

* ``MetaViewDA3Loader`` -- registers a ``depthanything3`` model folder (the
  community convention; legacy ``depth_anything_3`` is also scanned), then
  lazily loads the Depth-Anything-3 GIANT (3D-feature) model and the nested
  depth model into a ``DA3_MODEL`` bundle. Both HF-style model directories
  (containing config.json) and bare single-file checkpoints
  (e.g. ``da3_giant.safetensors``) are supported.
* ``MetaView3DConditioning`` -- runs DA3 on an input image to extract 3D
  features + a depth map, derives the target camera from yaw/pitch/radius
  (auto-radius from centre depth), and packages everything into the conditioning
  as ``metaview_*`` keys plus a correctly-sized sampler LATENT and (if a VAE is
  supplied) a reference latent of the edit image.

Contract (reconciled with docs/metaview_contract.md, the authoritative doc):

    metaview_viewmats   fp32  [2, 4, 4]      order (target, source=identity), world->cam
    metaview_ks         fp32  [2, 3, 3]      normalized intrinsics, both cams equal
    metaview_feat3d     bf16  [Hp, Wp, 6144] Hp=gen_h//16, Wp=gen_w//16, 4*1536
    metaview_depth      fp32  [2, gen_h, gen_w]  ch0 zeros, ch1 source depth (optional; see below)

Only these four keys are emitted. The batch dim ([b, n_cams, ...]) is omitted;
the model auto-promotes to b=1. viewmats/Ks stay float32 (the model bundles them
in a CONDConstant to bypass ComfyUI's bf16 cast) -- do NOT pre-cast them.

The reference/edit (source) image is VAE-encoded and attached as
``reference_latents`` (stock TextEncodeQwenImageEdit convention) at the target
latent resolution (gen_h/8 x gen_w/8); the model uses it as camera-1 tokens.

metaview_depth is OPTIONAL: the shipped model config uses prope_dim_arrange
[16,56,56] (no z-term), so depth does not change DiT outputs. It is still emitted
because it is contract-valid and meaningful upstream. It enters the DiT RAW (not
VAE-encoded).

Ks convention (RESOLVED): fx/(2*cx), fy/(2*cy) with principal-point entries 0,
consumed as-is by PRoPE — the reference-faithful convention this node emits.
The contract doc was corrected to match.
"""

import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

LOGGER = logging.getLogger(__name__)

# Layers whose intermediate features DA3-GIANT exports for the 3D feature volume.
EXPORT_3D_FEAT_LAYERS = [19, 27, 33, 39]
# PRoPE dimension arrangement used by the MetaView pipeline; length 4 => depth on.
PROPE_DIM_ARRANGE = [64, 20, 20, 24]
# Training pixel budget (960x528) the aspect rule area-matches against.
TARGET_AREA = 960 * 528

# ComfyUI model-folder key. DA3 models are discovered under every registered
# root for this key (typically ``<models>/depthanything3/``) as either HF-style
# subdirectories (config.json + model.safetensors) or bare *.safetensors files
# named after their preset (da3_giant.safetensors, da3nested_giant_large.safetensors, ...).
# The node NEVER creates junctions / hardlinks / symlinks: the user places (or
# points their model manager at) the DA3 models themselves.
_DA3_FOLDER_KEY = "depthanything3"
_DA3_LEGACY_DIRNAME = "depth_anything_3"
_FOLDER_REGISTERED = False


# ---------------------------------------------------------------------------
# Vendored DA3 access
# ---------------------------------------------------------------------------
def _get_da3_class():
    """Import the vendored DepthAnything3 class (see metaview_da3/)."""
    vendor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metaview_da3")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from depth_anything_3.api import DepthAnything3
    return DepthAnything3


# ---------------------------------------------------------------------------
# Pure geometry / dimension helpers (no ComfyUI dependency)
# ---------------------------------------------------------------------------
def compute_gen_dims(in_w, in_h, width_override=0, height_override=0):
    """Generation dims: aspect-preserving, area-matched to 960x528, multiple of 16.

    An explicit width & height override (both > 0) bypasses the aspect rule.
    """
    if width_override and height_override:
        return int(width_override), int(height_override)
    scale = (TARGET_AREA / (in_w * in_h)) ** 0.5
    gen_width = max(16, round(in_w * scale / 16) * 16)
    gen_height = max(16, round(in_h * scale / 16) * 16)
    return gen_width, gen_height


def compute_da3_process_res(gen_width, gen_height):
    """DA3 process_res coupling: max(gen)*14//16 (patch 14 vs DiT patch 16)."""
    return max(gen_width, gen_height) * 14 // 16


def compute_target_extrinsic(yaw_deg, pitch_deg, radius):
    """Port of inference_lowvram.compute_target_extrinsic (yaw/pitch/radius -> 4x4)."""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    R_y = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ])
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)],
    ])
    R = R_y @ R_x
    C = np.array([0.0, 0.0, radius])
    t = C - R @ C
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def image_tensor_to_pil(image):
    """ComfyUI IMAGE tensor [B,H,W,C] float 0..1 -> first-frame PIL RGB (uint8)."""
    if image.ndim == 4:
        image = image[0]
    arr = (image.clamp(0, 1).cpu().float().numpy() * 255.0).round().astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr[:, :, :3], "RGB")


# ---------------------------------------------------------------------------
# DA3 model bundle
# ---------------------------------------------------------------------------
class DA3ModelBundle:
    """Lazily-loaded pair of DA3 models (GIANT 3D-feature + nested depth).

    Models are loaded on first use and offloaded to CPU (freeing VRAM) after
    each inference so the DiT can claim the GPU next.
    """

    def __init__(self, giant_path, depth_path, device="cuda"):
        self.giant_path = giant_path
        self.depth_path = depth_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._giant = None
        self._depth = None

    # -- loading -----------------------------------------------------------
    def _load(self, path):
        DepthAnything3 = _get_da3_class()
        if os.path.isdir(path):
            return DepthAnything3.from_pretrained(path).eval()
        # Bare single-file checkpoint (community convention, e.g. da3_giant.safetensors):
        # build the network from the vendored MODEL_REGISTRY preset matching the
        # filename, then load the state dict. Key names are identical to the
        # HF-directory format; tied weights are recorded in safetensors metadata
        # as {duplicate_key: source_key} and must be re-materialized.
        preset = _da3_preset_from_filename(path)
        LOGGER.info("[MetaView] single-file DA3 checkpoint %s -> preset '%s'",
                    os.path.basename(path), preset)
        model = DepthAnything3(model_name=preset)
        from safetensors import safe_open
        from safetensors.torch import load_file
        sd = load_file(path)
        with safe_open(path, framework="pt") as f:
            tied = f.metadata() or {}
        for dup, src in tied.items():
            if dup not in sd and src in sd:
                sd[dup] = sd[src]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        real_missing = [k for k in missing if k not in tied]
        if real_missing or unexpected:
            raise RuntimeError(
                f"DA3 single-file load mismatch for {os.path.basename(path)} "
                f"(preset {preset}): {len(real_missing)} missing, "
                f"{len(unexpected)} unexpected keys. First missing: {real_missing[:3]}, "
                f"first unexpected: {list(unexpected)[:3]}")
        return model.eval()

    def giant_to(self, device):
        if self._giant is None:
            LOGGER.info("[MetaView] loading DA3-GIANT from %s", self.giant_path)
            self._giant = self._load(self.giant_path)
        self._giant.to(device=device)
        self._giant.device = torch.device(device)
        return self._giant

    def depth_to(self, device):
        if self._depth is None:
            LOGGER.info("[MetaView] loading DA3 depth model from %s", self.depth_path)
            self._depth = self._load(self.depth_path)
        self._depth.to(device=device)
        self._depth.device = torch.device(device)
        return self._depth

    def offload_giant(self):
        if self._giant is not None:
            self._giant.to(device="cpu")
            self._giant.device = torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def offload_depth(self):
        if self._depth is not None:
            self._depth.to(device="cpu")
            self._depth.device = torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Core DA3 conditioning extraction (pure; no ComfyUI dependency)
# ---------------------------------------------------------------------------
def run_da3_conditioning(
    bundle: DA3ModelBundle,
    pil_image: Image.Image,
    yaw,
    pitch,
    radius=0.0,
    auto_radius=True,
    width_override=0,
    height_override=0,
):
    """Replicate sections 1-3 of inference_lowvram.main() for one image.

    Returns a dict with viewmats [2,4,4] fp32, ks [2,3,3] fp32,
    feat3d [h16,w16,6144] bf16, depth [2,gh,gw] fp32, radius float, and gen dims.
    """
    device = bundle.device
    in_w, in_h = pil_image.size
    gen_width, gen_height = compute_gen_dims(in_w, in_h, width_override, height_override)
    edit_image = pil_image.resize((gen_width, gen_height))
    process_res = compute_da3_process_res(gen_width, gen_height)

    with torch.inference_mode():
        # --- 3D features + intrinsics (GIANT) ---
        model_3d = bundle.giant_to(device)
        feat_out = model_3d.inference(
            [edit_image],
            export_feat_layers=EXPORT_3D_FEAT_LAYERS,
            process_res=process_res,
        )
        # Normalized intrinsics, EXACTLY as the reference pipeline builds them
        # (inference_lowvram.py). fx is normalized by the DA3 sensor width 2*cx
        # (== process_res long side), fy by 2*cy, and the principal point is set
        # to 0. PRoPE._prepare_apply_fns consumes Ks AS-IS ("Ks has been
        # normalized in the dataset getitem"; PRoPE.py ~L279) -- it does NOT
        # re-normalize by image_width/height -- so these are the operative values.
        #
        # NOTE / CONTRACT CONFLICT: docs/metaview_contract.md 4 describes the
        # normalization as fx/gen_w with principal point ~0.5. That prose does
        # NOT match the working reference pipeline (which the parity test
        # confirms this code reproduces to kernel precision). Kept
        # reference-faithful deliberately; flagged for the model-side agent.
        intri = feat_out.intrinsics[0]
        width_i = intri[0, 2] * 2
        height_i = intri[1, 2] * 2
        ks_single = torch.Tensor([
            [intri[0, 0] / width_i, 0.0, 0.0],
            [0.0, intri[1, 1] / height_i, 0.0],
            [0.0, 0.0, 1.0],
        ])
        ks = torch.stack([ks_single, ks_single], dim=0)  # [2,3,3]

        feats = [
            torch.from_numpy(feat_out.aux[f"feat_layer_{layer}"])
            for layer in EXPORT_3D_FEAT_LAYERS
        ]
        feat_3d = torch.cat(feats, dim=-1).to(dtype=torch.bfloat16)  # [1,h_da3,w_da3,6144]
        feat_3d = feat_3d[0].contiguous().cpu()  # [h_da3,w_da3,6144]
        bundle.offload_giant()

        # --- depth (nested depth model) ---
        model_depth = bundle.depth_to(device)
        prediction = model_depth.inference([edit_image], process_res=process_res)
        depth_edit = torch.Tensor(prediction.depth).unsqueeze(0)  # [1,N,h,w]
        depth_edit = F.interpolate(
            depth_edit, size=(gen_height, gen_width), mode="bilinear", align_corners=False
        )[0]  # [N,gh,gw]
        depth_latent = torch.zeros_like(depth_edit)
        depth = torch.cat([depth_latent, depth_edit], dim=0)  # [2,gh,gw]
        bundle.offload_depth()

    depth = depth.float().cpu()

    # Contract 4: the 3D feature must sit on the DiT patch grid
    # (patches_y=gen_h//16, patches_x=gen_w//16); PRoPE asserts
    # seqlen == patches_x*patches_y. The process_res 14/16 coupling makes DA3's
    # native grid already equal to this for aspect-matched inputs (no-op resize,
    # preserving bit-exact parity), but guard for any rounding mismatch.
    patches_y, patches_x = gen_height // 16, gen_width // 16
    if feat_3d.shape[0] != patches_y or feat_3d.shape[1] != patches_x:
        LOGGER.info("[MetaView] resizing feat3d %s -> (%d,%d)",
                    tuple(feat_3d.shape[:2]), patches_y, patches_x)
        f = feat_3d.permute(2, 0, 1).unsqueeze(0).float()  # [1,6144,h,w]
        f = F.interpolate(f, size=(patches_y, patches_x), mode="bilinear", align_corners=False)
        feat_3d = f[0].permute(1, 2, 0).contiguous().to(torch.bfloat16)  # [Hp,Wp,6144]

    # --- radius ---
    if auto_radius or not radius:
        src = depth[1]
        radius = float(src[src.shape[0] // 2, src.shape[1] // 2].item())

    # --- target pose ---
    extrinsic_target = compute_target_extrinsic(yaw, pitch, radius)
    extrinsic_source = np.eye(4)
    viewmats = torch.Tensor(np.stack((extrinsic_target, extrinsic_source), axis=0)).float()  # [2,4,4]

    return {
        "viewmats": viewmats,
        "ks": ks.float(),
        "feat3d": feat_3d,
        "depth": depth,
        "radius": radius,
        "gen_width": gen_width,
        "gen_height": gen_height,
        "process_res": process_res,
        "edit_image": edit_image,
    }


# ---------------------------------------------------------------------------
# ComfyUI model-folder registration
# ---------------------------------------------------------------------------
def _register_da3_folder():
    """Register the ``depth_anything_3`` model folder with ComfyUI.

    Only registers the folder (and ensures the default root exists so the model
    manager has somewhere to place things). It does NOT create any junction,
    hardlink, or symlink -- the user manages the DA3 model directories with
    their own tooling. A missing/empty folder simply yields an empty dropdown.
    """
    global _FOLDER_REGISTERED
    if _FOLDER_REGISTERED:
        return
    try:
        import folder_paths
    except Exception:
        return  # not running inside ComfyUI (e.g. parity test)
    base = os.path.join(folder_paths.models_dir, _DA3_FOLDER_KEY)
    try:
        os.makedirs(base, exist_ok=True)
    except Exception as exc:  # never let folder setup break node import/listing
        LOGGER.warning("[MetaView] could not create %s: %s", base, exc)
    folder_paths.add_model_folder_path(_DA3_FOLDER_KEY, base)
    # Legacy folder name from earlier versions of this pack: scan it too, but
    # never create it.
    legacy = os.path.join(folder_paths.models_dir, _DA3_LEGACY_DIRNAME)
    if os.path.isdir(legacy):
        folder_paths.add_model_folder_path(_DA3_FOLDER_KEY, legacy)
    _FOLDER_REGISTERED = True


def _list_da3_subdirs():
    """DA3 entries under the registered folders.

    Lists both HF-style model directories (containing a config.json) and bare
    single-file ``*.safetensors`` checkpoints.
    """
    _register_da3_folder()
    try:
        import folder_paths
    except Exception:
        return []
    names = []
    for root in folder_paths.get_folder_paths(_DA3_FOLDER_KEY):
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            sub = os.path.join(root, name)
            is_model_dir = os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "config.json"))
            is_single_file = os.path.isfile(sub) and name.lower().endswith(".safetensors")
            if (is_model_dir or is_single_file) and name not in names:
                names.append(name)
    return names


def _resolve_da3_path(name):
    import folder_paths
    for root in folder_paths.get_folder_paths(_DA3_FOLDER_KEY):
        cand = os.path.join(root, name)
        if os.path.isdir(cand) or os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"DA3 model '{name}' not found under {_DA3_FOLDER_KEY} folders")


def _da3_preset_from_filename(path):
    """Map a bare checkpoint filename to a vendored DA3 MODEL_REGISTRY preset.

    ``da3_giant.safetensors`` -> ``da3-giant``,
    ``da3nested_giant_large.safetensors`` -> ``da3nested-giant-large`` etc.
    Tolerates case and trailing version suffixes (``da3_giant_1.1``).
    """
    from depth_anything_3.registry import MODEL_REGISTRY  # vendored
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    # strip trailing version-ish suffixes: "_1.1", "-v1", "_v1.0" ...
    import re
    stem = re.sub(r"[-_]v?\d+(\.\d+)*$", "", stem)
    candidate = stem.replace("_", "-")
    if candidate in MODEL_REGISTRY:
        return candidate
    raise ValueError(
        f"Unrecognized DA3 single-file checkpoint '{os.path.basename(path)}'. "
        f"Supported preset names: {sorted(MODEL_REGISTRY.keys())}. "
        "Rename the file to match a preset (e.g. da3_giant.safetensors) or use an "
        "HF-style model directory containing config.json.")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
class MetaViewDA3Loader:
    """Load the Depth-Anything-3 GIANT (3D features) + nested depth models."""

    @classmethod
    def INPUT_TYPES(cls):
        # Discovered DA3 model dirs under every registered depth_anything_3 root.
        # May be empty: the user places the DA3 model directories themselves (the
        # node creates no links). An empty list renders as an empty dropdown.
        choices = _list_da3_subdirs()
        loc_hint = ("Put Depth-Anything-3 models under models/depthanything3/ as either "
                    "single-file checkpoints (da3_giant.safetensors) or HF-style model "
                    "directories containing config.json (DA3-GIANT-1.1/).")
        return {
            "required": {
                "giant_model": (choices, {
                    "tooltip": "DA3 model providing 3D features + intrinsics (e.g. "
                               "da3_giant.safetensors or DA3-GIANT-1.1). " + loc_hint}),
                "depth_model": (choices, {
                    "tooltip": "DA3 nested model providing the depth map (e.g. "
                               "da3nested_giant_large.safetensors or "
                               "DA3NESTED-GIANT-LARGE-1.1). " + loc_hint}),
            }
        }

    RETURN_TYPES = ("DA3_MODEL",)
    RETURN_NAMES = ("da3_model",)
    FUNCTION = "load"
    CATEGORY = "EnviralDesign/MetaView"

    def load(self, giant_model, depth_model):
        giant_path = _resolve_da3_path(giant_model)
        depth_path = _resolve_da3_path(depth_model)
        bundle = DA3ModelBundle(giant_path, depth_path)
        return (bundle,)


class MetaView3DConditioning:
    """Augment Qwen-Image-Edit conditioning with MetaView 3D geometry.

    Runs DA3 on the image, computes the target camera from yaw/pitch/radius, and
    sets the metaview_* conditioning keys plus a sampler LATENT (and a reference
    latent if a VAE is supplied).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "da3_model": ("DA3_MODEL",),
                "image": ("IMAGE",),
                "yaw": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5,
                                  "tooltip": "Target-camera yaw in degrees."}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5,
                                    "tooltip": "Target-camera pitch in degrees."}),
                "radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                                     "tooltip": "Orbit radius. 0 or auto_radius => from centre depth."}),
                "auto_radius": ("BOOLEAN", {"default": True,
                                            "tooltip": "Derive radius from the centre-pixel depth."}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "Encodes the edit image into a reference latent."}),
                "width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16,
                                  "tooltip": "0 = auto from aspect rule."}),
                "height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16,
                                   "tooltip": "0 = auto from aspect rule."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("conditioning", "latent")
    FUNCTION = "build"
    CATEGORY = "EnviralDesign/MetaView"

    def build(self, conditioning, da3_model, image, yaw, pitch, radius, auto_radius,
              vae=None, width=0, height=0):
        import node_helpers

        pil_image = image_tensor_to_pil(image)
        out = run_da3_conditioning(
            da3_model, pil_image, yaw, pitch, radius, auto_radius,
            width_override=width, height_override=height,
        )
        gen_w, gen_h = out["gen_width"], out["gen_height"]

        # Four canonical keys. Model side owns prope_dim_arrange ([64,20,20,24]
        # from the executing demo) and derives the patch grid from the latent.
        values = {
            "metaview_viewmats": out["viewmats"],   # f32 [2,4,4] (target, source)
            "metaview_ks": out["ks"],               # f32 [2,3,3] normalized
            "metaview_feat3d": out["feat3d"],       # bf16 [Hp,Wp,6144]
            "metaview_depth": out["depth"],         # f32 [2,gen_h,gen_w] (optional)
        }
        conditioning = node_helpers.conditioning_set_values(conditioning, values)

        # Reference/edit latent (VAE-encoded edit image at gen dims). MetaView
        # feeds the DiT the edit latent at gen dims (no 1024x1024 auto-resize).
        if vae is not None:
            edit_np = np.asarray(out["edit_image"]).astype(np.float32) / 255.0
            edit_t = torch.from_numpy(edit_np).unsqueeze(0)  # [1,H,W,3]
            ref_latent = vae.encode(edit_t[:, :, :, :3])
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": [ref_latent]}, append=True
            )

        # Empty sampler latent sized to the gen dims (VAE /8 spatial, 16 ch).
        latent = {"samples": torch.zeros(1, 16, gen_h // 8, gen_w // 8)}
        return (conditioning, latent)


NODE_CLASS_MAPPINGS = {
    "MetaViewDA3Loader": MetaViewDA3Loader,
    "MetaView3DConditioning": MetaView3DConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MetaViewDA3Loader": "MetaView DA3 Loader",
    "MetaView3DConditioning": "MetaView 3D Conditioning",
}
