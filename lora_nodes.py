class _EnviralLoraLoaderBase:
    def __init__(self):
        self.loaded_lora = None

    @staticmethod
    def _lora_names():
        import folder_paths

        return folder_paths.get_filename_list("loras")

    @classmethod
    def _lora_name_input(cls):
        lora_names = cls._lora_names()
        default = lora_names[0] if lora_names else ""
        return (
            "STRING,COMBO",
            {
                "default": default,
                "widgetType": "COMBO",
                "options": lora_names,
                "tooltip": (
                    "LoRA name. Also accepts a linked STRING, as long as it "
                    "matches a LoRA path known to ComfyUI."
                ),
            },
        )

    @staticmethod
    def _resolve_lora_path(lora_name):
        import folder_paths

        lora_name = str(lora_name or "").strip()
        if not lora_name:
            raise ValueError("lora_name must not be empty")
        return folder_paths.get_full_path_or_raise("loras", lora_name)

    @classmethod
    def validate_inputs(cls, lora_name, **kwargs):
        if lora_name is None:
            return True
        try:
            cls._resolve_lora_path(lora_name)
        except Exception as err:
            return str(err)
        return True

    def _load_lora_file(self, lora_name):
        from comfy import utils

        lora_path = self._resolve_lora_path(lora_name)
        if self.loaded_lora is not None and self.loaded_lora[0] == lora_path:
            return self.loaded_lora[1], self.loaded_lora[2]

        lora, lora_metadata = utils.load_torch_file(
            lora_path,
            safe_load=True,
            return_metadata=True,
        )
        self.loaded_lora = (lora_path, lora, lora_metadata)
        return lora, lora_metadata


class EnviralLoadLora(_EnviralLoraLoaderBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "The diffusion model the LoRA will be applied to."},
                ),
                "clip": (
                    "CLIP",
                    {"tooltip": "The CLIP model the LoRA will be applied to."},
                ),
                "lora_name": cls._lora_name_input(),
                "strength_model": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the diffusion model.",
                    },
                ),
                "strength_clip": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the CLIP model.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"
    CATEGORY = "EnviralDesign/model"
    DESCRIPTION = """
Native-style LoRA loader with a string-linkable LoRA dropdown.
"""
    SEARCH_ALIASES = [
        "lora",
        "load lora",
        "apply lora",
        "lora loader",
        "lora model",
        "string lora",
    ]

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        from comfy import sd

        lora, lora_metadata = self._load_lora_file(lora_name)
        return sd.load_lora_for_models(
            model,
            clip,
            lora,
            strength_model,
            strength_clip,
            lora_metadata=lora_metadata,
        )


class EnviralLoadLoraModelOnly(_EnviralLoraLoaderBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "The diffusion model the LoRA will be applied to."},
                ),
                "lora_name": cls._lora_name_input(),
                "strength_model": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the diffusion model.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_lora_model_only"
    CATEGORY = "EnviralDesign/model"
    DESCRIPTION = """
Native-style model-only LoRA loader with a string-linkable LoRA dropdown.
"""
    SEARCH_ALIASES = [
        "lora",
        "load lora",
        "apply lora",
        "lora loader",
        "lora model only",
        "string lora",
    ]

    def load_lora_model_only(self, model, lora_name, strength_model):
        if strength_model == 0:
            return (model,)

        from comfy import sd

        lora, lora_metadata = self._load_lora_file(lora_name)
        model_lora, _ = sd.load_lora_for_models(
            model,
            None,
            lora,
            strength_model,
            0,
            lora_metadata=lora_metadata,
        )
        return (model_lora,)


NODE_CLASS_MAPPINGS = {
    "EnviralLoadLora": EnviralLoadLora,
    "EnviralLoadLoraModelOnly": EnviralLoadLoraModelOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnviralLoadLora": "Enviral Load LoRA",
    "EnviralLoadLoraModelOnly": "Enviral Load LoRA (Model Only)",
}
