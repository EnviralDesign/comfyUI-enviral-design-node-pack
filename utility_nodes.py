import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F


MAX_RESOLUTION = 16384
MISSING = object()


class LazyIndexSwitch:
    MAX_INPUTS = 20

    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {
            f"input_{index}": ("*", {"lazy": True})
            for index in range(cls.MAX_INPUTS)
        }
        return {
            "required": {
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": cls.MAX_INPUTS - 1,
                        "step": 1,
                        "tooltip": "Zero-based input index to select. This matches the INDEX output from ComfyUI combo nodes.",
                    },
                ),
            },
            "optional": optional_inputs,
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("selected",)
    FUNCTION = "select"
    CATEGORY = "EnviralDesign/utility"

    @classmethod
    def _input_name(cls, index):
        return f"input_{max(0, min(int(index), cls.MAX_INPUTS - 1))}"

    @classmethod
    def check_lazy_status(cls, index, **kwargs):
        input_name = cls._input_name(index)
        if kwargs.get(input_name, MISSING) is None:
            return [input_name]
        return []

    @classmethod
    def validate_inputs(cls, index, **kwargs):
        input_index = int(index)
        if input_index < 0 or input_index >= cls.MAX_INPUTS:
            return f"index must be between 0 and {cls.MAX_INPUTS - 1}"

        input_name = cls._input_name(input_index)
        if input_name not in kwargs:
            return f"{input_name} must be connected when index is {input_index}"

        return True

    def select(self, index, **kwargs):
        input_name = self._input_name(index)
        return (kwargs.get(input_name),)


NODE_CLASS_MAPPINGS = {
    "EnviralLazyIndexSwitch": LazyIndexSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnviralLazyIndexSwitch": "Lazy Index Switch",
}


class EnviralImageResizeKit:
    MODES = ["pass-through", "explicit resize", "inside fit", "outside fit"]
    MODE_ALIASES = {
        "pass": "pass-through",
        "passthrough": "pass-through",
        "pass through": "pass-through",
        "pass-through": "pass-through",
        "none": "pass-through",
        "source": "pass-through",
        "original": "pass-through",
        "explicit": "explicit resize",
        "explicit resize": "explicit resize",
        "resize": "explicit resize",
        "stretch": "explicit resize",
        "exact": "explicit resize",
        "inside": "inside fit",
        "inside fit": "inside fit",
        "fit inside": "inside fit",
        "contain": "inside fit",
        "letterbox": "inside fit",
        "pad": "inside fit",
        "outside": "outside fit",
        "outside fit": "outside fit",
        "fit outside": "outside fit",
        "cover": "outside fit",
        "crop": "outside fit",
        "center crop": "outside fit",
    }
    COLOR_NAMES = {
        "black": (0.0, 0.0, 0.0),
        "white": (1.0, 1.0, 1.0),
        "gray": (0.5, 0.5, 0.5),
        "grey": (0.5, 0.5, 0.5),
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (
                    "STRING,COMBO",
                    {
                        "default": "pass-through",
                        "widgetType": "COMBO",
                        "options": cls.MODES,
                        "tooltip": "Resize mode. Also accepts linked strings like crop, pad, resize, or pass-through.",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": MAX_RESOLUTION,
                        "step": 1,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": MAX_RESOLUTION,
                        "step": 1,
                    },
                ),
                "letterbox_color": (
                    "STRING",
                    {
                        "default": "0,0,0",
                        "tooltip": "Color used only for inside fit. Accepts R,G,B, 0-1 floats, hex, or simple names.",
                    },
                ),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK")
    RETURN_NAMES = ("image", "width", "height", "mask")
    FUNCTION = "resize"
    CATEGORY = "EnviralDesign/image"

    @classmethod
    def _normalize_mode(cls, mode):
        key = str(mode or "").strip().lower().replace("_", " ").replace("-", " ")
        key = re.sub(r"\s+", " ", key)
        compact_key = key.replace(" ", "")
        if key in cls.MODE_ALIASES:
            return cls.MODE_ALIASES[key]
        if compact_key in cls.MODE_ALIASES:
            return cls.MODE_ALIASES[compact_key]
        raise ValueError(
            f"Unsupported resize mode {mode!r}. Use one of: {', '.join(cls.MODES)}."
        )

    @classmethod
    def _parse_color(cls, color):
        value = str(color or "").strip().lower()
        if value in cls.COLOR_NAMES:
            return cls.COLOR_NAMES[value]

        hex_value = value
        if hex_value.startswith("#"):
            hex_value = hex_value[1:]
        if hex_value.startswith("0x"):
            hex_value = hex_value[2:]
        if re.fullmatch(r"[0-9a-f]{6}", hex_value):
            return tuple(int(hex_value[index:index + 2], 16) / 255.0 for index in (0, 2, 4))

        parts = [part for part in re.split(r"[\s,]+", value) if part]
        if len(parts) != 3:
            raise ValueError("letterbox_color must be R,G,B, #RRGGBB, or a simple color name.")

        numbers = [float(part) for part in parts]
        if any(number < 0 for number in numbers):
            raise ValueError("letterbox_color values must be non-negative.")
        if any(number > 1.0 for number in numbers):
            numbers = [number / 255.0 for number in numbers]
        return tuple(max(0.0, min(1.0, number)) for number in numbers)

    @staticmethod
    def _resize_image(image, width, height):
        if image.shape[1] == height and image.shape[2] == width:
            return image
        samples = image.movedim(-1, 1)
        resized = F.interpolate(samples, size=(height, width), mode="bicubic", align_corners=False)
        return resized.movedim(1, -1).clamp(0.0, 1.0)

    @staticmethod
    def _normalize_mask(mask, batch_size, height, width, image):
        if mask is None:
            return torch.zeros((batch_size, height, width), dtype=image.dtype, device=image.device)

        out_mask = mask.to(device=image.device, dtype=image.dtype)
        if out_mask.ndim == 2:
            out_mask = out_mask.unsqueeze(0)
        elif out_mask.ndim == 4:
            if out_mask.shape[-1] == 1:
                out_mask = out_mask[..., 0]
            elif out_mask.shape[1] == 1:
                out_mask = out_mask[:, 0]

        if out_mask.ndim != 3:
            raise ValueError(f"mask must be [height,width] or [batch,height,width], got {tuple(mask.shape)}")

        if out_mask.shape[0] == 1 and batch_size > 1:
            out_mask = out_mask.repeat(batch_size, 1, 1)
        elif out_mask.shape[0] != batch_size:
            raise ValueError(f"mask batch size {out_mask.shape[0]} does not match image batch size {batch_size}")

        if out_mask.shape[1] != height or out_mask.shape[2] != width:
            out_mask = EnviralImageResizeKit._resize_mask(out_mask, width, height)
        return out_mask.clamp(0.0, 1.0)

    @staticmethod
    def _resize_mask(mask, width, height):
        if mask.shape[1] == height and mask.shape[2] == width:
            return mask
        return F.interpolate(mask.unsqueeze(1), size=(height, width), mode="nearest-exact").squeeze(1)

    @staticmethod
    def _target_size(width, height):
        return max(1, int(width)), max(1, int(height))

    @classmethod
    def validate_inputs(cls, mode, width, height, letterbox_color, **kwargs):
        try:
            normalized_mode = cls._normalize_mode(mode)
            if normalized_mode != "pass-through":
                cls._target_size(width, height)
            if normalized_mode == "inside fit":
                cls._parse_color(letterbox_color)
        except Exception as err:
            return str(err)
        return True

    def resize(self, image, mode, width, height, letterbox_color, mask=None):
        batch_size, source_height, source_width, channels = image.shape
        normalized_mode = self._normalize_mode(mode)
        source_mask = self._normalize_mask(mask, batch_size, source_height, source_width, image)

        if normalized_mode == "pass-through":
            return (image, int(source_width), int(source_height), source_mask)

        target_width, target_height = self._target_size(width, height)

        if normalized_mode == "explicit resize":
            out_image = self._resize_image(image, target_width, target_height)
            out_mask = self._resize_mask(source_mask, target_width, target_height)
            return (out_image, target_width, target_height, out_mask)

        if normalized_mode == "inside fit":
            scale = min(target_width / source_width, target_height / source_height)
            scaled_width = max(1, round(source_width * scale))
            scaled_height = max(1, round(source_height * scale))
            scaled_image = self._resize_image(image, scaled_width, scaled_height)
            scaled_mask = self._resize_mask(source_mask, scaled_width, scaled_height)

            color = self._parse_color(letterbox_color)
            fill = list(color[:channels])
            if len(fill) < channels:
                fill.extend([1.0] * (channels - len(fill)))
            fill_tensor = image.new_tensor(fill).view(1, 1, 1, channels)

            out_image = image.new_empty((batch_size, target_height, target_width, channels))
            out_image[:] = fill_tensor
            out_mask = source_mask.new_zeros((batch_size, target_height, target_width))

            top = (target_height - scaled_height) // 2
            left = (target_width - scaled_width) // 2
            out_image[:, top:top + scaled_height, left:left + scaled_width, :] = scaled_image
            out_mask[:, top:top + scaled_height, left:left + scaled_width] = scaled_mask
            return (out_image, target_width, target_height, out_mask)

        if normalized_mode == "outside fit":
            scale = max(target_width / source_width, target_height / source_height)
            scaled_width = max(target_width, round(source_width * scale))
            scaled_height = max(target_height, round(source_height * scale))
            scaled_image = self._resize_image(image, scaled_width, scaled_height)
            scaled_mask = self._resize_mask(source_mask, scaled_width, scaled_height)

            top = (scaled_height - target_height) // 2
            left = (scaled_width - target_width) // 2
            out_image = scaled_image[:, top:top + target_height, left:left + target_width, :]
            out_mask = scaled_mask[:, top:top + target_height, left:left + target_width]
            return (out_image, target_width, target_height, out_mask)

        raise ValueError(f"Unsupported resize mode {mode!r}")


NODE_CLASS_MAPPINGS["EnviralImageResizeKit"] = EnviralImageResizeKit
NODE_DISPLAY_NAME_MAPPINGS["EnviralImageResizeKit"] = "Enviral Image Resize Kit"


class EnviralColorMatchV2:
    METHODS = [
        "mkl",
        "hm",
        "reinhard",
        "mvgd",
        "hm-mvgd-hm",
        "hm-mkl-hm",
        "reinhard_lab_gpu",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_target": ("IMAGE",),
                "image_ref": ("IMAGE",),
                "method": (cls.METHODS, {"default": "reinhard"}),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                    },
                ),
                "multithread": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "color_match"
    CATEGORY = "EnviralDesign/image"
    DESCRIPTION = """
Transfers color from image_ref to image_target using color-matcher methods, with
an optional Kornia Lab-space Reinhard path for GPU-friendly batch work.
"""

    @classmethod
    def validate_inputs(cls, method, strength, **kwargs):
        if method not in cls.METHODS:
            return f"method must be one of: {', '.join(cls.METHODS)}"
        if float(strength) < 0:
            return "strength must be non-negative"
        return True

    @staticmethod
    def _match_reference_batch(image_ref, batch_size):
        ref_batch_size = image_ref.shape[0]
        if ref_batch_size == batch_size:
            return image_ref
        if ref_batch_size > batch_size:
            return image_ref[:batch_size]

        indices = torch.arange(batch_size, device=image_ref.device).clamp_max(
            ref_batch_size - 1
        )
        return image_ref.index_select(0, indices)

    @classmethod
    def _reinhard_lab_gpu(cls, image_target, image_ref, strength):
        if image_target.shape[-1] != 3 or image_ref.shape[-1] != 3:
            raise ValueError("reinhard_lab_gpu requires RGB IMAGE tensors with 3 channels.")

        try:
            import kornia
            from comfy import model_management
        except ImportError as err:
            raise ImportError(
                "reinhard_lab_gpu requires ComfyUI's kornia dependency to be available."
            ) from err

        device = model_management.get_torch_device()
        batch_size = image_target.shape[0]
        matched_ref = cls._match_reference_batch(image_ref, batch_size)

        target_bchw = (
            image_target.to(device=device, dtype=torch.float32)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        ref_bchw = (
            matched_ref.to(device=device, dtype=torch.float32)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        target_lab = kornia.color.rgb_to_lab(target_bchw)
        ref_lab = kornia.color.rgb_to_lab(ref_bchw)

        target_flat = target_lab.flatten(start_dim=2)
        ref_flat = ref_lab.flatten(start_dim=2)

        target_std, target_mean = torch.std_mean(target_flat, dim=-1, keepdim=True, unbiased=False)
        ref_std, ref_mean = torch.std_mean(ref_flat, dim=-1, keepdim=True, unbiased=False)
        target_std = target_std.clamp_min(1e-6)

        corrected_flat = (target_flat - target_mean) * (ref_std / target_std) + ref_mean
        corrected_lab = corrected_flat.view_as(target_lab)
        corrected_rgb = kornia.color.lab_to_rgb(corrected_lab)

        out = (1.0 - strength) * target_bchw + strength * corrected_rgb
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.cpu().float().clamp_(0.0, 1.0)

    @classmethod
    def _color_matcher_transfer(cls, image_target, image_ref, method, strength, multithread):
        try:
            from color_matcher import ColorMatcher
        except ImportError as err:
            raise ImportError(
                "Color Match V2 requires color-matcher. "
                "Install this node pack's declared dependencies."
            ) from err

        batch_size = image_target.shape[0]
        ref_batch_size = image_ref.shape[0]

        def process(index):
            color_matcher = ColorMatcher()
            target_np = image_target[index].detach().cpu().numpy()
            ref_np = image_ref[min(index, ref_batch_size - 1)].detach().cpu().numpy()

            try:
                matched_np = color_matcher.transfer(src=target_np, ref=ref_np, method=method)
                if strength != 1.0:
                    matched_np = target_np + strength * (matched_np - target_np)
                return torch.from_numpy(matched_np)
            except Exception as err:
                logging.warning("Color Match V2 item %s failed: %s", index, err)
                return torch.from_numpy(target_np)

        if bool(multithread) and batch_size > 1:
            max_workers = min(os.cpu_count() or 1, batch_size)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                out = list(executor.map(process, range(batch_size)))
        else:
            out = [process(index) for index in range(batch_size)]

        return torch.stack(out, dim=0).to(dtype=torch.float32).clamp_(0.0, 1.0)

    def color_match(self, image_target, image_ref, method, strength=1.0, multithread=True):
        strength = float(strength)
        if strength == 0.0:
            return (image_target,)

        if method == "reinhard_lab_gpu":
            return (self._reinhard_lab_gpu(image_target, image_ref, strength),)

        return (
            self._color_matcher_transfer(
                image_target, image_ref, method, strength, multithread
            ),
        )


NODE_CLASS_MAPPINGS["EnviralColorMatchV2"] = EnviralColorMatchV2
NODE_DISPLAY_NAME_MAPPINGS["EnviralColorMatchV2"] = "Enviral Color Match V2"


class EnviralModelPatchTorchSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_fp16_accumulation": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Toggles torch.backends.cuda.matmul.allow_fp16_accumulation "
                            "while the patched model runs."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "EnviralDesign/model"
    DESCRIPTION = """
Adds model callbacks that enable or disable PyTorch full FP16 accumulation for
CUDA FP16 matmuls while the model runs.
"""
    EXPERIMENTAL = True

    @staticmethod
    def _ensure_supported():
        if not hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
            raise RuntimeError(
                "torch.backends.cuda.matmul.allow_fp16_accumulation is not "
                "available in this PyTorch build."
            )

    @staticmethod
    def _set_fp16_accumulation(enabled):
        logging.info(
            "Patching torch settings: "
            "torch.backends.cuda.matmul.allow_fp16_accumulation = %s",
            enabled,
        )
        torch.backends.cuda.matmul.allow_fp16_accumulation = enabled

    def patch(self, model, enable_fp16_accumulation):
        from comfy.patcher_extension import CallbacksMP

        self._ensure_supported()
        if not hasattr(model, "clone") or not hasattr(model, "add_callback"):
            raise RuntimeError(
                "Model Patch Torch Settings requires a ComfyUI MODEL patcher "
                "with clone() and add_callback() support."
            )

        model_clone = model.clone()

        def enable_callback(model_patcher):
            self._set_fp16_accumulation(True)

        def disable_callback(model_patcher):
            self._set_fp16_accumulation(False)

        if enable_fp16_accumulation:
            model_clone.add_callback(CallbacksMP.ON_PRE_RUN, enable_callback)
            model_clone.add_callback(CallbacksMP.ON_CLEANUP, disable_callback)
        else:
            model_clone.add_callback(CallbacksMP.ON_PRE_RUN, disable_callback)

        return (model_clone,)


NODE_CLASS_MAPPINGS["EnviralModelPatchTorchSettings"] = EnviralModelPatchTorchSettings
NODE_DISPLAY_NAME_MAPPINGS["EnviralModelPatchTorchSettings"] = "Model Patch Torch Settings"
