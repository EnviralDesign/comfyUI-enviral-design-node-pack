"""MetaView DiT for ComfyUI.

A grafted Qwen-Image-Edit transformer for novel-view synthesis. The base transformer is
Qwen-Image-Edit (diffusers/comfy naming); the graft adds:
  * ``_3D_in``: Linear(6144 -> 3072) projecting a DA3 3D feature into the transformer width.
  * per-block ``prope_attn`` (MetaViewSelfAttention3D): a parallel self-attention branch that
    jointly attends image tokens + 3D-feature tokens, both PRoPE positional-encoded; its output
    is added to the main image attention output.
  * PRoPE rotary encoding (Cameras as Relative Positional Encoding).

The math is ported faithfully from ``C:\\repos\\MetaView\\src\\MetaView_dit.py`` and
``MetaView_pipeline.py`` (``model_fn_MetaView``), but built on comfy ``operations`` so weights
stream/cast through comfy's native machinery and fp8 layers execute on comfy's quantized op path.

See ``docs/metaview_contract.md`` for the conditioning contract.
"""

import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import comfy.ops
import comfy.conds
import comfy.utils
import comfy.model_base
import comfy.model_patcher
import comfy.model_management
import comfy.supported_models_base
import comfy.latent_formats
from comfy.model_base import ModelType
from comfy.ldm.qwen_image.model import (
    QwenImageTransformer2DModel,
    QwenTimestepProjEmbeddings,
    LastLayer,
)

try:  # loaded as part of the node pack (package-relative)
    from .metaview.prope import PropeDotProductAttention
    from .metaview.qwen_rope import QwenEmbedRope, apply_rotary_emb_qwen, approximate_gelu
except ImportError:  # loaded as a top-level module (standalone tests)
    from metaview.prope import PropeDotProductAttention
    from metaview.qwen_rope import QwenEmbedRope, apply_rotary_emb_qwen, approximate_gelu

try:
    import folder_paths
except ImportError:  # not available outside comfy (unit tests that don't need the node)
    folder_paths = None

LOGGER = logging.getLogger(__name__)

# Fixed architecture constants for this artifact (metaview_fp8_scaled_v1).
_DIM = 3072
_NUM_HEADS = 24
_HEAD_DIM = 128
_NUM_LAYERS = 60
_JOINT_DIM = 3584          # Qwen2.5-VL text hidden dim (txt_norm / txt_in input)
_3D_DIM = 6144             # DA3 feature channel dim (_3D_in input)
_ADD_IN_DIM = 3072         # projected 3D-feature width
_PATCH = 2
_IN_CHANNELS = 64          # 16 latent channels * patch(2) * patch(2)
_OUT_CHANNELS = 16
# PRoPE dimension arrangement. The MetaView reference demo (src/inference.py:86,
# src/inference_lowvram.py) uses the 4-term arrangement [64, 20, 20, 24] with the
# DEPTH channel active (len == 4 -> depth term), and feeds the DA3 depth map into
# PRoPE precompute. NOTE: MetaViewPipeline's signature default [16, 56, 56] is
# overridden by the demo and is NOT what the released weights were trained with.
_PROPE_DIM_ARRANGE = [64, 20, 20, 24]
_PROPE_FREQ_BASE = 10000.0


# ---------------------------------------------------------------------------
# attention helper (matches diffsynth qwen_image_flash_attention non-flash path)
# ---------------------------------------------------------------------------
def _attention(q, k, v, num_heads, attn_mask=None):
    # q,k,v: [b, heads, seq, head_dim]
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return rearrange(x, "b n s d -> b s (n d)", n=num_heads)


# ---------------------------------------------------------------------------
# feed forward (diffsynth QwenFeedForward: ApproximateGELU = x*sigmoid(1.702x))
# ---------------------------------------------------------------------------
class _ProjGELU(nn.Module):
    """FeedForward.net[0] — holds the ``.proj`` Linear then the sigmoid-GELU activation."""

    def __init__(self, dim_in, dim_out, dtype, device, operations):
        super().__init__()
        self.proj = operations.Linear(dim_in, dim_out, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        return approximate_gelu(self.proj(x))


class MetaViewFeedForward(nn.Module):
    def __init__(self, dim, dim_out, dtype, device, operations):
        super().__init__()
        inner = dim * 4
        self.net = nn.ModuleList([
            _ProjGELU(dim, inner, dtype, device, operations),          # net.0 (.proj)
            nn.Dropout(0.0),                                           # net.1
            operations.Linear(inner, dim_out, bias=True, dtype=dtype, device=device),  # net.2
        ])

    def forward(self, x):
        for m in self.net:
            x = m(x)
        return x


# ---------------------------------------------------------------------------
# main double-stream attention (diffsynth QwenDoubleStreamAttention)
# In MetaView's shipped config this branch is invoked with prope=None, so image tokens receive
# the standard Qwen 2D RoPE (via image_rotary_emb) exactly like base Qwen-Image-Edit.
# ---------------------------------------------------------------------------
class QwenDoubleStreamAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, dtype, device, operations):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.to_k = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.to_v = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.norm_q = operations.RMSNorm(head_dim, eps=1e-6, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(head_dim, eps=1e-6, elementwise_affine=True, dtype=dtype, device=device)

        self.add_q_proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.add_k_proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.add_v_proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.norm_added_q = operations.RMSNorm(head_dim, eps=1e-6, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_added_k = operations.RMSNorm(head_dim, eps=1e-6, elementwise_affine=True, dtype=dtype, device=device)

        self.to_out = nn.Sequential(operations.Linear(dim, dim, dtype=dtype, device=device))
        self.to_add_out = operations.Linear(dim, dim, dtype=dtype, device=device)

    def forward(self, image, text, image_rotary_emb=None, attention_mask=None, prope=None):
        img_q, img_k, img_v = self.to_q(image), self.to_k(image), self.to_v(image)
        txt_q, txt_k, txt_v = self.add_q_proj(text), self.add_k_proj(text), self.add_v_proj(text)
        seq_txt = txt_q.shape[1]

        img_q = rearrange(img_q, "b s (h d) -> b h s d", h=self.num_heads)
        img_k = rearrange(img_k, "b s (h d) -> b h s d", h=self.num_heads)
        img_v = rearrange(img_v, "b s (h d) -> b h s d", h=self.num_heads)
        txt_q = rearrange(txt_q, "b s (h d) -> b h s d", h=self.num_heads)
        txt_k = rearrange(txt_k, "b s (h d) -> b h s d", h=self.num_heads)
        txt_v = rearrange(txt_v, "b s (h d) -> b h s d", h=self.num_heads)

        img_q, img_k = self.norm_q(img_q), self.norm_k(img_k)
        txt_q, txt_k = self.norm_added_q(txt_q), self.norm_added_k(txt_k)

        if prope is not None and image_rotary_emb is not None:
            _, txt_freqs = image_rotary_emb
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs)
            img_q = prope._apply_to_q(img_q)
            img_k = prope._apply_to_kv(img_k)
        elif image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_q = apply_rotary_emb_qwen(img_q, img_freqs)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs)

        joint_q = torch.cat([txt_q, img_q], dim=2)
        joint_k = torch.cat([txt_k, img_k], dim=2)
        joint_v = torch.cat([txt_v, img_v], dim=2)

        out = _attention(joint_q, joint_k, joint_v, self.num_heads, attn_mask=attention_mask).to(joint_q.dtype)
        txt_out = out[:, :seq_txt, :]
        img_out = out[:, seq_txt:, :]

        img_out = self.to_out[0](img_out)
        txt_out = self.to_add_out(txt_out)
        return img_out, txt_out


# ---------------------------------------------------------------------------
# grafted parallel 3D attention (diffsynth MetaViewSelfAttention3D)
# ---------------------------------------------------------------------------
class MetaViewSelfAttention3D(nn.Module):
    def __init__(self, dim_a, dim_b, num_heads, head_dim, merge_3D, dtype, device, operations):
        super().__init__()
        self.merge_3D = merge_3D
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = operations.Linear(dim_a, dim_a, dtype=dtype, device=device)
        self.to_k = operations.Linear(dim_a, dim_a, dtype=dtype, device=device)
        self.to_v = operations.Linear(dim_a, dim_a, dtype=dtype, device=device)
        self.to_q_3D = operations.Linear(dim_b, dim_a, dtype=dtype, device=device)
        self.to_k_3D = operations.Linear(dim_b, dim_a, dtype=dtype, device=device)
        self.to_v_3D = operations.Linear(dim_b, dim_a, dtype=dtype, device=device)

        self.norm_q = operations.RMSNorm(head_dim, eps=1e-6, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(head_dim, eps=1e-6, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_added_q = operations.RMSNorm(head_dim, eps=1e-6, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_added_k = operations.RMSNorm(head_dim, eps=1e-6, elementwise_affine=True, dtype=dtype, device=device)

        self.to_out = nn.Sequential(operations.Linear(dim_a, dim_a, dtype=dtype, device=device))

    def forward(self, image, feat_3D, attention_mask=None, prope=None, add_prope=None):
        img_q, img_k, img_v = self.to_q(image), self.to_k(image), self.to_v(image)
        _3D_q, _3D_k, _3D_v = self.to_q_3D(feat_3D), self.to_k_3D(feat_3D), self.to_v_3D(feat_3D)
        seq_img = img_q.shape[1]

        img_q = rearrange(img_q, "b s (h d) -> b h s d", h=self.num_heads)
        img_k = rearrange(img_k, "b s (h d) -> b h s d", h=self.num_heads)
        img_v = rearrange(img_v, "b s (h d) -> b h s d", h=self.num_heads)
        img_q, img_k = self.norm_q(img_q), self.norm_k(img_k)

        _3D_q = rearrange(_3D_q, "b s (h d) -> b h s d", h=self.num_heads)
        _3D_k = rearrange(_3D_k, "b s (h d) -> b h s d", h=self.num_heads)
        _3D_v = rearrange(_3D_v, "b s (h d) -> b h s d", h=self.num_heads)
        _3D_q, _3D_k = self.norm_added_q(_3D_q), self.norm_added_k(_3D_k)

        if prope is not None:
            img_q = prope._apply_to_q(img_q)
            img_k = prope._apply_to_kv(img_k)
            img_v = prope._apply_to_kv(img_v)
            _3D_q = add_prope._apply_to_q(_3D_q)
            _3D_k = add_prope._apply_to_kv(_3D_k)
            _3D_v = add_prope._apply_to_kv(_3D_v)

        joint_q = torch.cat([img_q, _3D_q], dim=2)
        joint_k = torch.cat([img_k, _3D_k], dim=2)
        joint_v = torch.cat([img_v, _3D_v], dim=2)

        num_heads = img_q.shape[1]
        out = _attention(joint_q, joint_k, joint_v, num_heads, attn_mask=attention_mask).to(img_q.dtype)

        img_out = out[:, :seq_img, :]
        _3D_out = out[:, seq_img:, :]

        if prope is not None:
            img_out = rearrange(img_out, "b s (n d) -> b n s d", n=num_heads)
            img_out = prope._apply_to_o(img_out)
            img_out = rearrange(img_out, "b n s d -> b s (n d)", n=num_heads)
        img_out = self.to_out(img_out)

        if self.merge_3D:
            if add_prope is not None:
                _3D_out = rearrange(_3D_out, "b s (n d) -> b n s d", n=num_heads)
                _3D_out = add_prope._apply_to_o(_3D_out)
                _3D_out = rearrange(_3D_out, "b n s d -> b s (n d)", n=num_heads)
            _3D_out = self.to_out(_3D_out)
            return img_out, _3D_out
        return img_out, None


# ---------------------------------------------------------------------------
# grafted transformer block (diffsynth MetaViewTransformerBlock, merge_3D=True)
# ---------------------------------------------------------------------------
class MetaViewTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, add_in_dim, merge_3D, eps, dtype, device, operations):
        super().__init__()
        self.merge_3D = merge_3D

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, bias=True, dtype=dtype, device=device),
        )
        self.img_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.attn = QwenDoubleStreamAttention(dim, num_heads, head_dim, dtype, device, operations)
        self.prope_attn = MetaViewSelfAttention3D(dim, add_in_dim, num_heads, head_dim, merge_3D, dtype, device, operations)

        self.img_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_mlp = MetaViewFeedForward(dim, dim, dtype, device, operations)

        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, bias=True, dtype=dtype, device=device),
        )
        self.txt_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.txt_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.txt_mlp = MetaViewFeedForward(dim, dim, dtype, device, operations)

    @staticmethod
    def _modulate(x, mod_params):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(self, image, text, temb, image_rotary_emb=None, attention_mask=None,
                prope=None, add_prope=None, add_attn=True, feat_3D=None):
        img_mod_attn, img_mod_mlp = self.img_mod(temb).chunk(2, dim=-1)
        txt_mod_attn, txt_mod_mlp = self.txt_mod(temb).chunk(2, dim=-1)

        img_normed = self.img_norm1(image)
        img_modulated, img_gate = self._modulate(img_normed, img_mod_attn)

        txt_normed = self.txt_norm1(text)
        txt_modulated, txt_gate = self._modulate(txt_normed, txt_mod_attn)

        if self.merge_3D:
            feat_3D_modulated = self.img_norm1(feat_3D)
        else:
            feat_3D_modulated = feat_3D

        _3D_prope_out = None
        if add_attn and prope is not None:
            img_prope_out, _3D_prope_out = self.prope_attn(
                image=img_modulated, feat_3D=feat_3D_modulated,
                attention_mask=attention_mask, prope=prope, add_prope=add_prope,
            )
            img_attn_out, txt_attn_out = self.attn(
                image=img_modulated, text=txt_modulated,
                image_rotary_emb=image_rotary_emb, attention_mask=attention_mask, prope=None,
            )
            img_attn_out = img_attn_out + img_prope_out
        else:
            img_attn_out, txt_attn_out = self.attn(
                image=img_modulated, text=txt_modulated,
                image_rotary_emb=image_rotary_emb, attention_mask=attention_mask, prope=prope,
            )

        image = image + img_gate * img_attn_out
        text = text + txt_gate * txt_attn_out

        img_normed_2 = self.img_norm2(image)
        img_modulated_2, img_gate_2 = self._modulate(img_normed_2, img_mod_mlp)
        txt_normed_2 = self.txt_norm2(text)
        txt_modulated_2, txt_gate_2 = self._modulate(txt_normed_2, txt_mod_mlp)

        image = image + img_gate_2 * self.img_mlp(img_modulated_2)
        text = text + txt_gate_2 * self.txt_mlp(txt_modulated_2)

        if self.merge_3D:
            feat_3D = feat_3D + _3D_prope_out
            feat_3D = feat_3D + self.img_mlp(self.img_norm2(feat_3D))
            return text, image, feat_3D
        return text, image, feat_3D


# ---------------------------------------------------------------------------
# full grafted DiT
# ---------------------------------------------------------------------------
class MetaViewDiT(QwenImageTransformer2DModel):
    """Grafted Qwen-Image-Edit transformer. Subclasses comfy's QwenImageTransformer2DModel to
    inherit its forward()/patcher-wrapper plumbing, but rebuilds the module tree with the
    MetaView graft and overrides _forward with the MetaView denoise math."""

    def __init__(self, image_model=None, dtype=None, device=None, operations=None, **kwargs):
        nn.Module.__init__(self)
        self.dtype = dtype
        self.patch_size = _PATCH
        self.in_channels = _IN_CHANNELS
        self.out_channels = _OUT_CHANNELS
        self.inner_dim = _DIM
        self.num_heads = _NUM_HEADS
        self.head_dim = _HEAD_DIM
        self.merge_3D = True

        # Text-stream RoPE (image stream is handled by PRoPE / Qwen rope via this too).
        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)

        # Base Qwen-Image-Edit modules (names match the artifact keys exactly).
        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=768,
            dtype=dtype, device=device, operations=operations)
        self.txt_norm = operations.RMSNorm(_JOINT_DIM, eps=1e-6, dtype=dtype, device=device)
        self.img_in = operations.Linear(_IN_CHANNELS, self.inner_dim, dtype=dtype, device=device)
        self.txt_in = operations.Linear(_JOINT_DIM, self.inner_dim, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList([
            MetaViewTransformerBlock(
                dim=self.inner_dim, num_heads=self.num_heads, head_dim=self.head_dim,
                add_in_dim=_ADD_IN_DIM, merge_3D=True, eps=1e-6,
                dtype=dtype, device=device, operations=operations)
            for _ in range(_NUM_LAYERS)
        ])

        self.norm_out = LastLayer(self.inner_dim, self.inner_dim, dtype=dtype, device=device, operations=operations)
        self.proj_out = operations.Linear(self.inner_dim, _PATCH * _PATCH * self.out_channels, bias=True, dtype=dtype, device=device)

        # Grafted 3D-feature input projection.
        self._3D_in = operations.Linear(_3D_DIM, _ADD_IN_DIM, dtype=dtype, device=device)

    # ---- helpers -----------------------------------------------------------
    def _patchify(self, latent):
        # latent [b, 16, H_lat, W_lat] -> tokens [b, (H_lat/2 * W_lat/2), 64]
        H = latent.shape[2] // _PATCH
        W = latent.shape[3] // _PATCH
        return rearrange(latent, "b c (H P) (W Q) -> b (H W) (c P Q)", H=H, W=W, P=_PATCH, Q=_PATCH), H, W

    def _build_prope(self, patches_x, patches_y, viewmats, ks, depth=None):
        p = PropeDotProductAttention(
            head_dim=self.head_dim, patches_x=patches_x, patches_y=patches_y,
            image_width=patches_x * 16, image_height=patches_y * 16,
            freq_base=_PROPE_FREQ_BASE, dim_arrange=_PROPE_DIM_ARRANGE,
        ).to(viewmats.device)
        p._precompute_and_cache_apply_fns(viewmats, ks, depth)
        return p

    # ---- main forward ------------------------------------------------------
    def _forward(self, x, timesteps, context, attention_mask=None, ref_latents=None,
                 additional_t_cond=None, transformer_options={}, control=None,
                 metaview=None, **kwargs):
        orig_ndim = x.ndim
        if orig_ndim == 5:  # comfy passes [b, C, T=1, H, W] for image models
            x = x[:, :, 0]

        image, patches_y, patches_x = self._patchify(x)
        image_seq_len = image.shape[1]
        img_shapes = [(1, patches_y, patches_x)]

        if ref_latents is not None and len(ref_latents) > 0:
            edit = ref_latents[0]
            if edit.ndim == 5:
                edit = edit[:, :, 0]
            edit_tokens, e_H, e_W = self._patchify(edit.to(x.dtype))
            image = torch.cat([image, edit_tokens], dim=1)
            img_shapes.append((1, e_H, e_W))

        # text sequence lengths
        if attention_mask is not None:
            txt_seq_lens = attention_mask.sum(dim=1).tolist()
        else:
            txt_seq_lens = [context.shape[1]] * context.shape[0]

        image = self.img_in(image)
        temb = self.time_text_embed(timesteps, image, additional_t_cond)
        text = self.txt_in(self.txt_norm(context))

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=x.device)

        # ---- build PRoPE from cameras -------------------------------------
        prope = None
        add_prope = None
        feat_3D = None
        if metaview is not None and metaview.get("viewmats") is not None and metaview.get("ks") is not None:
            viewmats = metaview["viewmats"]
            ks = metaview["ks"]
            if viewmats.ndim == 3:
                viewmats = viewmats.unsqueeze(0)
            if ks.ndim == 3:
                ks = ks.unsqueeze(0)
            viewmats = viewmats.to(device=x.device, dtype=torch.float32)
            ks = ks.to(device=x.device, dtype=torch.float32)

            # Depth enters PRoPE (dim_arrange has 4 terms -> depth channel active),
            # bilinearly resized to the token grid, exactly like the reference
            # (MetaView_pipeline.py model_fn: F.interpolate to (h//16, w//16)).
            depth = metaview.get("depth")
            if depth is not None:
                if depth.ndim == 3:  # [n_cams, H, W] -> [1, n_cams, H, W]
                    depth = depth.unsqueeze(0)
                depth = depth.to(device=x.device, dtype=torch.float32)
                depth = F.interpolate(depth, size=(patches_y, patches_x),
                                      mode="bilinear", align_corners=False)

            prope = self._build_prope(patches_x, patches_y, viewmats, ks, depth)

            feat = metaview.get("feat3d")
            if feat is not None:
                if feat.ndim == 3:
                    feat = feat.unsqueeze(0)
                feat = feat.to(device=x.device, dtype=image.dtype)
                feat_3D = rearrange(feat, "b h w d -> b (h w) d")
                feat_3D = self._3D_in(feat_3D)
                # add_PRoPE uses the source camera only (index 1), incl. its depth
                add_prope = self._build_prope(
                    patches_x, patches_y, viewmats[:, 1:2], ks[:, 1:2],
                    depth[:, 1:2] if depth is not None else None)

        use_graft = prope is not None and feat_3D is not None

        for block in self.transformer_blocks:
            text, image, feat_3D = block(
                image=image, text=text, temb=temb,
                image_rotary_emb=image_rotary_emb, attention_mask=None,
                prope=prope, add_prope=add_prope, add_attn=use_graft, feat_3D=feat_3D,
            )

        image = self.norm_out(image, temb)
        image = self.proj_out(image)
        image = image[:, :image_seq_len]

        out = rearrange(image, "b (H W) (c P Q) -> b c (H P) (W Q)",
                        H=patches_y, W=patches_x, P=_PATCH, Q=_PATCH)
        if orig_ndim == 5:
            out = out.unsqueeze(2)
        return out


# ---------------------------------------------------------------------------
# state-dict preparation: synthesize comfy_quant markers for fp8 layers
# ---------------------------------------------------------------------------
def prepare_metaview_state_dict(sd):
    """The artifact stores fp8 weights + ``<layer>.weight_scale`` companions but no per-layer
    ``comfy_quant`` markers. Comfy's modern quantized load path (MixedPrecisionOps) keys on the
    marker, so synthesize one for every fp8 layer. Values are untouched (marker only).

    ``full_precision_matrix_mult: true`` matches Comfy-Org's scaled-fp8 Qwen convention and the
    MetaView reference ``--no_fp8_mm`` path: dequant weight to bf16, run F.linear. Without it,
    MixedPrecisionOps quantizes activations to fp8 every matmul and the denoise trajectory
    diverges into noise relative to the working reference.
    """
    marker = torch.tensor(
        list(json.dumps({
            "format": "float8_e4m3fn",
            "full_precision_matrix_mult": True,
        }).encode("utf-8")),
        dtype=torch.uint8,
    )
    fp8_layers = []
    for k, v in sd.items():
        if k.endswith(".weight") and getattr(v, "dtype", None) == torch.float8_e4m3fn:
            layer = k[: -len(".weight")]
            if f"{layer}.weight_scale" in sd:
                fp8_layers.append(layer)
    for layer in fp8_layers:
        sd[f"{layer}.comfy_quant"] = marker.clone()
    return sd, len(fp8_layers)


# ---------------------------------------------------------------------------
# comfy model config + BaseModel subclass
# ---------------------------------------------------------------------------
# Default mu for the training pixel budget 960x528 → patch grid 60x33 = 1980 tokens.
# diffsynth FlowMatchScheduler._calculate_shift_qwen_image(1980) ≈ 0.5869.
_QWEN_DEFAULT_MU = (0.9 - 0.5) / (8192 - 256) * 1980 + (0.5 - (0.9 - 0.5) / (8192 - 256) * 256)
_QWEN_SHIFT_TERMINAL = 0.02


def calculate_qwen_image_mu(image_seq_len, base_seq_len=256, max_seq_len=8192,
                            base_shift=0.5, max_shift=0.9):
    """Match diffsynth FlowMatchScheduler._calculate_shift_qwen_image."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return float(image_seq_len * m + b)


def qwen_image_sigmas(steps, mu=_QWEN_DEFAULT_MU, shift_terminal=_QWEN_SHIFT_TERMINAL):
    """Exact diffsynth Qwen-Image FlowMatch schedule (incl. terminal shift to 0.02).

    Pack-local helper — used by the MetaViewSigmas node. Does not register into
    Comfy's global scheduler table.
    """
    import math
    sigmas = torch.linspace(1.0, 0.0, int(steps) + 1)[:-1]
    sigmas = math.exp(mu) / (math.exp(mu) + (1.0 / sigmas - 1.0))
    one_minus_z = 1.0 - sigmas
    scale_factor = one_minus_z[-1] / (1.0 - shift_terminal)
    sigmas = 1.0 - (one_minus_z / scale_factor)
    return torch.cat([sigmas, sigmas.new_zeros(1)])


class MetaViewModel(comfy.model_base.QwenImage):
    """Reuses QwenImage's ref-latent/attention-mask extra_conds and adds the MetaView bundle."""

    def __init__(self, model_config, model_type=ModelType.FLUX, device=None):
        # ModelType.FLUX, exactly like comfy's native model_base.QwenImage: the sampler feeds
        # sigma in [0,1] and QwenTimestepProjEmbeddings' Timesteps(scale=1000) applies the
        # x1000 internally (comfy/ldm/qwen_image/model.py). The reference MetaView_pipeline is
        # equivalent: it divides its [0,1000] timesteps by 1000 before its own scale=1000
        # embedder. ModelType.FLOW would double-scale the timestep and break denoising.
        #
        # Pair with MetaViewSigmas (exact Qwen-Image schedule) + stock SamplerCustom; do not
        # rely on KSampler's named schedulers for the terminal-shift schedule.
        comfy.model_base.BaseModel.__init__(
            self, model_config, model_type, device=device, unet_model=MetaViewDiT)
        self.memory_usage_factor_conds = ("ref_latents",)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        mv = {}
        for src, dst in (("metaview_viewmats", "viewmats"),
                         ("metaview_ks", "ks"),
                         ("metaview_feat3d", "feat3d"),
                         ("metaview_depth", "depth")):
            v = kwargs.get(src, None)
            if v is not None:
                mv[dst] = v
        if mv:
            # CONDConstant (a plain dict) bypasses _apply_model's automatic bf16 cast so the
            # float32 camera matrices survive intact.
            out["metaview"] = comfy.conds.CONDConstant(mv)
        return out


class MetaViewConfig(comfy.supported_models_base.BASE):
    unet_config = {"image_model": "metaview"}
    unet_extra_config = {}
    # Default mu for 960x528; MetaViewSigmas recomputes from width/height when provided.
    sampling_settings = {"multiplier": 1.0, "shift": _QWEN_DEFAULT_MU}
    memory_usage_factor = 1.8
    latent_format = comfy.latent_formats.Wan21
    supported_inference_dtypes = [torch.bfloat16, torch.float32]
    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        return MetaViewModel(self, device=device)


def _metaview_unet_config():
    return {"image_model": "metaview"}


def load_metaview_model(ckpt_path):
    """Load the MetaView artifact and return a ComfyUI ModelPatcher (MODEL)."""
    sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    sd, n_fp8 = prepare_metaview_state_dict(sd)
    LOGGER.info("MetaView: synthesized comfy_quant markers for %d fp8 layers", n_fp8)

    parameters = comfy.utils.calculate_parameters(sd)
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()

    model_config = MetaViewConfig(_metaview_unet_config())
    model_config.quant_config = {"mixed_ops": True}  # routes pick_operations -> mixed_precision_ops

    unet_dtype = comfy.model_management.unet_dtype(
        model_params=parameters,
        supported_dtypes=model_config.supported_inference_dtypes,
        weight_dtype=None,
    )
    manual_cast_dtype = comfy.model_management.unet_manual_cast(
        None, load_device, model_config.supported_inference_dtypes)
    if manual_cast_dtype is None:
        manual_cast_dtype = torch.bfloat16  # ensure a real compute dtype for MixedPrecisionOps
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    model = model_config.get_model(sd, "")
    model_patcher = comfy.model_patcher.ModelPatcher(
        model, load_device=load_device, offload_device=offload_device)
    if not comfy.model_management.is_device_cpu(offload_device):
        model.to(offload_device)
    model.load_model_weights(sd, "")
    left = list(sd.keys())
    if left:
        LOGGER.warning("MetaView: %d leftover state-dict keys after load: %s", len(left), left[:8])
    return model_patcher


# ---------------------------------------------------------------------------
# loader + schedule nodes (pack-local; no Comfy globals mutated)
# ---------------------------------------------------------------------------
class MetaViewModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("diffusion_models") if folder_paths else []
        return {"required": {"model_name": (models,)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "EnviralDesign/MetaView"
    TITLE = "MetaView Model Loader"

    def load(self, model_name):
        path = folder_paths.get_full_path("diffusion_models", model_name)
        if path is None:
            raise FileNotFoundError(f"MetaView model '{model_name}' not found in diffusion_models")
        return (load_metaview_model(path),)


class MetaViewSigmas:
    """Emit the exact Qwen-Image / MetaView FlowMatch sigma schedule as SIGMAS.

    Wire into stock ``SamplerCustom`` / ``SamplerCustomAdvanced`` (not KSampler's
    named-scheduler dropdown). Pass the MetaView3DConditioning LATENT so mu is
    derived from the real generation grid; width/height widgets are fallbacks.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
            },
            "optional": {
                "latent": ("LATENT", {"tooltip": "Prefer this: mu from latent HxW (*8 → pixels)."}),
                "width": ("INT", {"default": 960, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 528, "min": 16, "max": 8192, "step": 16}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "build"
    CATEGORY = "EnviralDesign/MetaView"
    TITLE = "MetaView Sigmas"

    def build(self, steps, latent=None, width=960, height=528):
        if latent is not None and "samples" in latent:
            # Wan/Qwen VAE: spatial /8. Patch grid for mu is /16 → latent_h/2 * latent_w/2.
            samples = latent["samples"]
            lat_h, lat_w = int(samples.shape[-2]), int(samples.shape[-1])
            height, width = lat_h * 8, lat_w * 8
        mu = calculate_qwen_image_mu((height // 16) * (width // 16))
        return (qwen_image_sigmas(steps, mu=mu),)


NODE_CLASS_MAPPINGS = {
    "MetaViewModelLoader": MetaViewModelLoader,
    "MetaViewSigmas": MetaViewSigmas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MetaViewModelLoader": "MetaView Model Loader",
    "MetaViewSigmas": "MetaView Sigmas",
}
