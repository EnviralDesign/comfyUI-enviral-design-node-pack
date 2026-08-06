ALL_LORA_FOLDERS = "All LoRAs"
MAX_LORA_BANKS = 5


def _normalize_lora_path(value):
    return str(value or "").strip().replace("\\", "/").strip("/")


def _lora_folder_prefixes(lora_names):
    prefixes = set()
    for lora_name in lora_names:
        parts = _normalize_lora_path(lora_name).split("/")[:-1]
        prefixes.update("/".join(parts[:index]) for index in range(1, len(parts) + 1))
    return [ALL_LORA_FOLDERS, *sorted(prefixes, key=str.casefold)]


def _lora_is_in_folder(lora_name, folder):
    if folder == ALL_LORA_FOLDERS:
        return True
    normalized_name = _normalize_lora_path(lora_name).casefold()
    normalized_folder = _normalize_lora_path(folder).casefold()
    return bool(normalized_folder) and normalized_name.startswith(normalized_folder + "/")


def _lora_display_name(lora_name, folder):
    normalized_name = _normalize_lora_path(lora_name)
    if folder == ALL_LORA_FOLDERS:
        return normalized_name
    normalized_folder = _normalize_lora_path(folder)
    return normalized_name[len(normalized_folder) + 1 :]


def _parse_lora_allow_list(allow_list):
    return {
        _normalize_lora_path(line).casefold()
        for line in str(allow_list or "").splitlines()
        if _normalize_lora_path(line)
    }


def _filtered_lora_names(lora_names, folder, allow_list=""):
    allowed = _parse_lora_allow_list(allow_list)
    filtered = []
    for lora_name in lora_names:
        if not _lora_is_in_folder(lora_name, folder):
            continue
        display_name = _lora_display_name(lora_name, folder)
        if allowed and not {
            _normalize_lora_path(lora_name).casefold(),
            _normalize_lora_path(display_name).casefold(),
        } & allowed:
            continue
        filtered.append((lora_name, display_name))
    return filtered


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

    @classmethod
    def _folder_input(cls):
        return (
            "STRING,COMBO",
            {
                "default": ALL_LORA_FOLDERS,
                "widgetType": "COMBO",
                "options": _lora_folder_prefixes(cls._lora_names()),
                "tooltip": "Limits the LoRA dropdown to this folder and its subfolders.",
            },
        )

    @classmethod
    def _resolve_folder(cls, folder):
        folder = str(folder or "").strip()
        folders = _lora_folder_prefixes(cls._lora_names())
        if folder in folders:
            return folder

        normalized_folder = _normalize_lora_path(folder).casefold()
        matches = [
            candidate
            for candidate in folders
            if _normalize_lora_path(candidate).casefold() == normalized_folder
        ]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"folder value {folder!r} must be one of the current LoRA folders")

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
    DEPRECATED = True

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


class EnviralLoadLoraFiltered(EnviralLoadLora):
    DEPRECATED = False

    @staticmethod
    def _bank_input_name(input_name, bank_index):
        if bank_index == 1:
            return input_name
        return f"{input_name}_{bank_index}"

    @classmethod
    def _bank_values(
        cls,
        lora_name,
        strength_model,
        strength_clip,
        bank_count,
        kwargs,
    ):
        try:
            bank_count = int(bank_count)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"bank_count must be between 1 and {MAX_LORA_BANKS}"
            ) from err
        if not 1 <= bank_count <= MAX_LORA_BANKS:
            raise ValueError(f"bank_count must be between 1 and {MAX_LORA_BANKS}")

        banks = [(1, lora_name, strength_model, strength_clip)]
        for bank_index in range(2, bank_count + 1):
            banks.append(
                (
                    bank_index,
                    kwargs.get(cls._bank_input_name("lora_name", bank_index)),
                    kwargs.get(cls._bank_input_name("strength_model", bank_index), 0.0)
                    or 0.0,
                    kwargs.get(cls._bank_input_name("strength_clip", bank_index), 0.0)
                    or 0.0,
                )
            )
        return banks

    @staticmethod
    def _is_bypassed(strength_model, strength_clip):
        return strength_model == 0 and strength_clip == 0

    @classmethod
    def _validate_lora_bank(cls, folder, lora_name, bank_index):
        if not str(lora_name or "").strip():
            raise ValueError(
                f"LoRA bank {bank_index} must select a LoRA when either strength is non-zero"
            )
        cls._resolve_lora_path(lora_name)
        if not _lora_is_in_folder(lora_name, folder):
            raise ValueError(f"LoRA {lora_name!r} is not inside folder {folder!r}")

    @classmethod
    def _available_loras(cls, folder, allow_list=""):
        return _filtered_lora_names(cls._lora_names(), folder, allow_list)

    @classmethod
    def INPUT_TYPES(cls):
        required = super().INPUT_TYPES()["required"]
        optional = {
            "clip": required["clip"],
            "allow_list": (
                "STRING",
                {
                    "forceInput": True,
                    "multiline": True,
                    "tooltip": (
                        "Optional newline-separated LoRA allow-list. Connect ComfyUI's "
                        "Text (Multiline) node; entries may use displayed names or full paths."
                    ),
                },
            ),
        }
        for bank_index in range(2, MAX_LORA_BANKS + 1):
            for input_name in ("lora_name", "strength_model", "strength_clip"):
                input_type, options = required[input_name]
                optional[cls._bank_input_name(input_name, bank_index)] = (
                    input_type,
                    {
                        **options,
                        "default": 0.0 if input_name.startswith("strength_") else options["default"],
                        "display_name": f"{input_name} {bank_index}",
                    },
                )
        return {
            "required": {
                "model": required["model"],
                "folder": cls._folder_input(),
                "bank_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": MAX_LORA_BANKS,
                        "display_name": "banks",
                        "tooltip": "Number of LoRA banks shown and applied.",
                    },
                ),
                "lora_name": (
                    required["lora_name"][0],
                    {**required["lora_name"][1], "display_name": "lora_name 1"},
                ),
                "strength_model": (
                    required["strength_model"][0],
                    {
                        **required["strength_model"][1],
                        "display_name": "strength_model 1",
                    },
                ),
                "strength_clip": (
                    required["strength_clip"][0],
                    {
                        **required["strength_clip"][1],
                        "display_name": "strength_clip 1",
                    },
                ),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "lora_list")
    OUTPUT_TOOLTIPS = (
        "The modified diffusion model.",
        "The modified CLIP model.",
        "The visible LoRA options, one display name per line.",
    )
    FUNCTION = "load_lora_filtered"
    DESCRIPTION = """
LoRA loader with folder and text allow-list filters plus independently bypassable banks.
"""
    SEARCH_ALIASES = [
        "lora",
        "load lora",
        "filtered lora",
        "folder lora",
        "curated lora",
    ]

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        folder,
        lora_name,
        strength_model=1.0,
        strength_clip=1.0,
        bank_count=1,
        allow_list="",
        **kwargs,
    ):
        try:
            folder = cls._resolve_folder(folder)
            available_loras = {
                lora_name for lora_name, _ in cls._available_loras(folder, allow_list)
            }
            banks = cls._bank_values(
                lora_name,
                strength_model,
                strength_clip,
                bank_count,
                kwargs,
            )
            active_banks = [
                bank
                for bank in banks
                if not cls._is_bypassed(bank[2], bank[3])
            ]
            if not active_banks:
                return True

            for bank_index, lora_name, _, _ in active_banks:
                cls._validate_lora_bank(folder, lora_name, bank_index)
                if lora_name not in available_loras:
                    raise ValueError(
                        f"LoRA {lora_name!r} is not included by the current allow-list"
                    )
        except Exception as err:
            return str(err)
        return True

    @classmethod
    def validate_inputs(cls, folder, lora_name, **kwargs):
        return cls.VALIDATE_INPUTS(folder, lora_name, **kwargs)

    def load_lora_filtered(
        self,
        model,
        folder,
        lora_name,
        strength_model,
        strength_clip,
        bank_count=1,
        clip=None,
        allow_list="",
        **kwargs,
    ):
        folder = self._resolve_folder(folder)
        available_loras = self._available_loras(folder, allow_list)
        available_paths = {lora_name for lora_name, _ in available_loras}
        lora_list = "\n".join(display_name for _, display_name in available_loras)
        banks = self._bank_values(
            lora_name,
            strength_model,
            strength_clip,
            bank_count,
            kwargs,
        )
        if clip is None:
            banks = [
                (bank_index, lora_name, strength_model, 0.0)
                for bank_index, lora_name, strength_model, _ in banks
            ]
        active_banks = [
            bank for bank in banks if not self._is_bypassed(bank[2], bank[3])
        ]
        if not active_banks:
            return (model, clip, lora_list)

        for bank_index, lora_name, strength_model, strength_clip in active_banks:
            self._validate_lora_bank(folder, lora_name, bank_index)
            if lora_name not in available_paths:
                raise ValueError(
                    f"LoRA {lora_name!r} is not included by the current allow-list"
                )
            model, clip = self.load_lora(
                model, clip, lora_name, strength_model, strength_clip
            )
        return (model, clip, lora_list)


class EnviralLoadLoraModelOnly(_EnviralLoraLoaderBase):
    DEPRECATED = True

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
    "EnviralLoadLoraFiltered": EnviralLoadLoraFiltered,
    "EnviralLoadLoraModelOnly": EnviralLoadLoraModelOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnviralLoadLora": "Enviral Load LoRA (Deprecated)",
    "EnviralLoadLoraFiltered": "Enviral Load LoRA",
    "EnviralLoadLoraModelOnly": "Enviral Load LoRA (Model Only, Deprecated)",
}
