import torch


ALL_CHECKPOINT_FOLDERS = "All Checkpoints"
ALL_DIFFUSION_MODEL_FOLDERS = "All Diffusion Models"
ALL_TEXT_ENCODER_FOLDERS = "All Text Encoders"

DIFFUSION_MODEL_WEIGHT_DTYPES = [
    "default",
    "fp8_e4m3fn",
    "fp8_e4m3fn_fast",
    "fp8_e5m2",
]

TEXT_ENCODER_TYPES = [
    "stable_diffusion",
    "stable_cascade",
    "sd3",
    "stable_audio",
    "mochi",
    "ltxv",
    "pixart",
    "cosmos",
    "lumina2",
    "wan",
    "hidream",
    "chroma",
    "ace",
    "omnigen2",
    "qwen_image",
    "hunyuan_image",
    "flux2",
    "ovis",
    "longcat_image",
    "cogvideox",
    "lens",
    "pixeldit",
    "ideogram4",
    "boogu",
    "krea2",
    "joyimage",
    "mage",
    "minimax",
]


def _normalize_path(value):
    return str(value or "").strip().replace("\\", "/").strip("/")


def _folder_prefixes(file_names, all_folders):
    prefixes = set()
    for file_name in file_names:
        parts = _normalize_path(file_name).split("/")[:-1]
        prefixes.update("/".join(parts[:index]) for index in range(1, len(parts) + 1))
    return [all_folders, *sorted(prefixes, key=str.casefold)]


def _is_in_folder(file_name, folder, all_folders):
    if folder == all_folders:
        return True
    normalized_name = _normalize_path(file_name).casefold()
    normalized_folder = _normalize_path(folder).casefold()
    return bool(normalized_folder) and normalized_name.startswith(normalized_folder + "/")


def _display_name(file_name, folder, all_folders):
    normalized_name = _normalize_path(file_name)
    if folder == all_folders:
        return normalized_name
    normalized_folder = _normalize_path(folder)
    return normalized_name[len(normalized_folder) + 1 :]


def _parse_allow_list(allow_list):
    return {
        _normalize_path(line).casefold()
        for line in str(allow_list or "").splitlines()
        if _normalize_path(line)
    }


class _EnviralFilteredModelLoaderBase:
    FOLDER_CATEGORY = None
    ALL_FOLDERS = None
    FILE_INPUT_NAME = None

    @classmethod
    def _file_names(cls):
        import folder_paths

        return folder_paths.get_filename_list(cls.FOLDER_CATEGORY)

    @classmethod
    def _folder_input(cls):
        return (
            "STRING,COMBO",
            {
                "default": cls.ALL_FOLDERS,
                "widgetType": "COMBO",
                "options": _folder_prefixes(cls._file_names(), cls.ALL_FOLDERS),
                "tooltip": "Limits the model dropdown to this folder and its subfolders.",
            },
        )

    @classmethod
    def _file_name_input(cls):
        file_names = cls._file_names()
        default = file_names[0] if file_names else ""
        return (
            "STRING,COMBO",
            {
                "default": default,
                "widgetType": "COMBO",
                "options": file_names,
                "tooltip": (
                    "Model name. Also accepts a linked STRING, as long as it "
                    "matches a model path known to ComfyUI."
                ),
            },
        )

    @classmethod
    def _resolve_folder(cls, folder):
        folder = str(folder or "").strip()
        folders = _folder_prefixes(cls._file_names(), cls.ALL_FOLDERS)
        if folder in folders:
            return folder

        normalized_folder = _normalize_path(folder).casefold()
        matches = [
            candidate
            for candidate in folders
            if _normalize_path(candidate).casefold() == normalized_folder
        ]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(
            f"folder value {folder!r} must be one of the current {cls.FOLDER_CATEGORY} folders"
        )

    @classmethod
    def _resolve_file_path(cls, file_name):
        import folder_paths

        file_name = str(file_name or "").strip()
        if not file_name:
            raise ValueError(f"{cls.FILE_INPUT_NAME} must not be empty")
        return folder_paths.get_full_path_or_raise(cls.FOLDER_CATEGORY, file_name)

    @classmethod
    def _available_files(cls, folder, allow_list=""):
        allowed = _parse_allow_list(allow_list)
        filtered = []
        for file_name in cls._file_names():
            if not _is_in_folder(file_name, folder, cls.ALL_FOLDERS):
                continue
            display_name = _display_name(file_name, folder, cls.ALL_FOLDERS)
            if allowed and not {
                _normalize_path(file_name).casefold(),
                _normalize_path(display_name).casefold(),
            } & allowed:
                continue
            filtered.append((file_name, display_name))
        return filtered

    @classmethod
    def _validate_selection(cls, folder, file_name, allow_list=""):
        folder = cls._resolve_folder(folder)
        cls._resolve_file_path(file_name)
        if not _is_in_folder(file_name, folder, cls.ALL_FOLDERS):
            raise ValueError(f"{cls.FILE_INPUT_NAME} {file_name!r} is not inside folder {folder!r}")
        available_files = {file_name for file_name, _ in cls._available_files(folder, allow_list)}
        if file_name not in available_files:
            raise ValueError(
                f"{cls.FILE_INPUT_NAME} {file_name!r} is not included by the current allow-list"
            )
        return folder

    @classmethod
    def VALIDATE_INPUTS(cls, folder, allow_list="", **kwargs):
        try:
            cls._validate_selection(folder, kwargs.get(cls.FILE_INPUT_NAME), allow_list)
        except Exception as err:
            return str(err)
        return True

    @classmethod
    def validate_inputs(cls, folder, **kwargs):
        return cls.VALIDATE_INPUTS(folder, **kwargs)

    @classmethod
    def _visible_file_list(cls, folder, allow_list):
        return "\n".join(
            display_name for _, display_name in cls._available_files(folder, allow_list)
        )


class EnviralLoadCheckpointFiltered(_EnviralFilteredModelLoaderBase):
    FOLDER_CATEGORY = "checkpoints"
    ALL_FOLDERS = ALL_CHECKPOINT_FOLDERS
    FILE_INPUT_NAME = "ckpt_name"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": cls._folder_input(),
                "ckpt_name": cls._file_name_input(),
            },
            "optional": {
                "allow_list": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": True,
                        "tooltip": (
                            "Optional newline-separated checkpoint allow-list. Entries may "
                            "use displayed names or full paths."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "checkpoint_list")
    OUTPUT_TOOLTIPS = (
        "The diffusion model loaded from the checkpoint.",
        "The text encoder loaded from the checkpoint.",
        "The VAE loaded from the checkpoint.",
        "The visible checkpoint options, one display name per line.",
    )
    FUNCTION = "load_checkpoint"
    CATEGORY = "EnviralDesign/model"
    DESCRIPTION = "Checkpoint loader with folder and text allow-list filters."
    SEARCH_ALIASES = ["checkpoint", "load checkpoint", "filtered checkpoint", "model loader"]

    def load_checkpoint(self, folder, ckpt_name, allow_list=""):
        folder = self._validate_selection(folder, ckpt_name, allow_list)
        ckpt_path = self._resolve_file_path(ckpt_name)

        import folder_paths
        from comfy import sd

        model, clip, vae, *_ = sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return model, clip, vae, self._visible_file_list(folder, allow_list)


class EnviralLoadDiffusionModelFiltered(_EnviralFilteredModelLoaderBase):
    FOLDER_CATEGORY = "diffusion_models"
    ALL_FOLDERS = ALL_DIFFUSION_MODEL_FOLDERS
    FILE_INPUT_NAME = "unet_name"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": cls._folder_input(),
                "unet_name": cls._file_name_input(),
                "weight_dtype": (
                    DIFFUSION_MODEL_WEIGHT_DTYPES,
                    {"default": "default", "advanced": True},
                ),
            },
            "optional": {
                "allow_list": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": True,
                        "tooltip": (
                            "Optional newline-separated diffusion-model allow-list. Entries "
                            "may use displayed names or full paths."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "diffusion_model_list")
    OUTPUT_TOOLTIPS = (
        "The loaded diffusion model.",
        "The visible diffusion-model options, one display name per line.",
    )
    FUNCTION = "load_diffusion_model"
    CATEGORY = "EnviralDesign/model"
    DESCRIPTION = "Diffusion-model loader with folder and text allow-list filters."
    SEARCH_ALIASES = ["diffusion model", "load unet", "filtered unet", "unet loader"]

    def load_diffusion_model(self, folder, unet_name, weight_dtype, allow_list=""):
        folder = self._validate_selection(folder, unet_name, allow_list)
        model_options = {}

        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        from comfy import sd

        model = sd.load_diffusion_model(
            self._resolve_file_path(unet_name),
            model_options=model_options,
        )
        return model, self._visible_file_list(folder, allow_list)


class EnviralLoadTextEncoderFiltered(_EnviralFilteredModelLoaderBase):
    FOLDER_CATEGORY = "text_encoders"
    ALL_FOLDERS = ALL_TEXT_ENCODER_FOLDERS
    FILE_INPUT_NAME = "clip_name"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": cls._folder_input(),
                "clip_name": cls._file_name_input(),
                "type": (TEXT_ENCODER_TYPES, {"default": "stable_diffusion"}),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
                "allow_list": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": True,
                        "tooltip": (
                            "Optional newline-separated text-encoder allow-list. Entries may "
                            "use displayed names or full paths."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "text_encoder_list")
    OUTPUT_TOOLTIPS = (
        "The loaded text encoder.",
        "The visible text-encoder options, one display name per line.",
    )
    FUNCTION = "load_text_encoder"
    CATEGORY = "EnviralDesign/model"
    DESCRIPTION = "Text-encoder loader with folder and text allow-list filters."
    SEARCH_ALIASES = ["text encoder", "load clip", "filtered clip", "clip loader"]

    def load_text_encoder(self, folder, clip_name, type="stable_diffusion", device="default", allow_list=""):
        folder = self._validate_selection(folder, clip_name, allow_list)
        model_options = {}

        from comfy import sd

        clip_type = getattr(sd.CLIPType, type.upper(), sd.CLIPType.STABLE_DIFFUSION)
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        import folder_paths

        clip = sd.load_clip(
            ckpt_paths=[self._resolve_file_path(clip_name)],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options,
        )
        return clip, self._visible_file_list(folder, allow_list)


NODE_CLASS_MAPPINGS = {
    "EnviralLoadCheckpointFiltered": EnviralLoadCheckpointFiltered,
    "EnviralLoadDiffusionModelFiltered": EnviralLoadDiffusionModelFiltered,
    "EnviralLoadTextEncoderFiltered": EnviralLoadTextEncoderFiltered,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnviralLoadCheckpointFiltered": "Enviral Load Checkpoint",
    "EnviralLoadDiffusionModelFiltered": "Enviral Load Diffusion Model",
    "EnviralLoadTextEncoderFiltered": "Enviral Load Text Encoder",
}
