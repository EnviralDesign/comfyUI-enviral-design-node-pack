class _ExactComboOutput(list):
    def __init__(self, options_provider):
        self._options_provider = options_provider
        super().__init__(self._current_options())

    def _current_options(self):
        return list(self._options_provider() or [])

    def __eq__(self, other):
        if other == "*":
            return True
        if isinstance(other, list):
            return list(other) == self._current_options()
        return list.__eq__(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)


def _sampler_names():
    import comfy.samplers

    return comfy.samplers.KSampler.SAMPLERS


def _scheduler_names():
    import comfy.samplers

    return comfy.samplers.KSampler.SCHEDULERS


def _checkpoint_names():
    import folder_paths

    return folder_paths.get_filename_list("checkpoints")


def _vae_names():
    import nodes

    return nodes.VAELoader.vae_list(nodes.VAELoader)


def _lora_names():
    import folder_paths

    return folder_paths.get_filename_list("loras")


def _color_match_methods():
    return [
        "mkl",
        "hm",
        "reinhard",
        "mvgd",
        "hm-mvgd-hm",
        "hm-mkl-hm",
    ]


def _color_match_v2_methods():
    return _color_match_methods() + ["reinhard_lab_gpu"]


class _NameSelectorBase:
    INPUT_NAME = "name"
    DISPLAY_LABEL = "Name"
    OPTIONS_PROVIDER = staticmethod(lambda: [])
    CATEGORY = "EnviralDesign/selectors"
    FUNCTION = "select"

    @classmethod
    def _options(cls):
        return list(cls.OPTIONS_PROVIDER() or [])

    @classmethod
    def _input(cls):
        options = cls._options()
        default = options[0] if options else ""
        return (
            "STRING,COMBO",
            {
                "default": default,
                "widgetType": "COMBO",
                "options": options,
                "tooltip": (
                    f"{cls.DISPLAY_LABEL}. Accepts a linked string, then "
                    "outputs the exact native combo type for downstream nodes."
                ),
            },
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {cls.INPUT_NAME: cls._input()}}

    @classmethod
    def _resolve_choice(cls, value):
        value = str(value or "").strip()
        options = cls._options()
        if not value:
            raise ValueError(f"{cls.INPUT_NAME} must not be empty")
        if value not in options:
            raise ValueError(
                f"{cls.INPUT_NAME} must be one of the current ComfyUI options"
            )
        return value

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        if kwargs.get(cls.INPUT_NAME) is None:
            return True
        try:
            cls._resolve_choice(kwargs.get(cls.INPUT_NAME))
        except Exception as err:
            return str(err)
        return True

    @classmethod
    def validate_inputs(cls, **kwargs):
        return cls.VALIDATE_INPUTS(**kwargs)

    def select(self, **kwargs):
        return (self._resolve_choice(kwargs.get(self.INPUT_NAME)),)


class EnviralSamplerName(_NameSelectorBase):
    INPUT_NAME = "sampler_name"
    DISPLAY_LABEL = "Sampler name"
    OPTIONS_PROVIDER = staticmethod(_sampler_names)
    RETURN_TYPES = (_ExactComboOutput(_sampler_names),)
    RETURN_NAMES = ("sampler_name",)
    OUTPUT_TOOLTIPS = ("Native-compatible sampler name combo.",)
    DESCRIPTION = """
Selects or adapts a sampler name for native sampler_name combo inputs.
"""
    SEARCH_ALIASES = ["sampler", "sampler name", "ksampler sampler"]


class EnviralSchedulerName(_NameSelectorBase):
    INPUT_NAME = "scheduler"
    DISPLAY_LABEL = "Scheduler"
    OPTIONS_PROVIDER = staticmethod(_scheduler_names)
    RETURN_TYPES = (_ExactComboOutput(_scheduler_names),)
    RETURN_NAMES = ("scheduler",)
    OUTPUT_TOOLTIPS = ("Native-compatible scheduler combo.",)
    DESCRIPTION = """
Selects or adapts a scheduler name for native scheduler combo inputs.
"""
    SEARCH_ALIASES = ["scheduler", "scheduler name", "ksampler scheduler"]


class EnviralCheckpointName(_NameSelectorBase):
    INPUT_NAME = "ckpt_name"
    DISPLAY_LABEL = "Checkpoint name"
    OPTIONS_PROVIDER = staticmethod(_checkpoint_names)
    RETURN_TYPES = (_ExactComboOutput(_checkpoint_names),)
    RETURN_NAMES = ("ckpt_name",)
    OUTPUT_TOOLTIPS = ("Native-compatible checkpoint name combo.",)
    DESCRIPTION = """
Selects or adapts a checkpoint name for native ckpt_name combo inputs.
"""
    SEARCH_ALIASES = ["checkpoint", "ckpt", "model name", "checkpoint name"]


class EnviralVAEName(_NameSelectorBase):
    INPUT_NAME = "vae_name"
    DISPLAY_LABEL = "VAE name"
    OPTIONS_PROVIDER = staticmethod(_vae_names)
    RETURN_TYPES = (_ExactComboOutput(_vae_names),)
    RETURN_NAMES = ("vae_name",)
    OUTPUT_TOOLTIPS = ("Native-compatible VAE name combo.",)
    DESCRIPTION = """
Selects or adapts a VAE name for native vae_name combo inputs.
"""
    SEARCH_ALIASES = ["vae", "vae name", "load vae"]


class EnviralLoraName(_NameSelectorBase):
    INPUT_NAME = "lora_name"
    DISPLAY_LABEL = "LoRA name"
    OPTIONS_PROVIDER = staticmethod(_lora_names)
    RETURN_TYPES = (_ExactComboOutput(_lora_names),)
    RETURN_NAMES = ("lora_name",)
    OUTPUT_TOOLTIPS = ("Native-compatible LoRA name combo.",)
    DESCRIPTION = """
Selects or adapts a LoRA name for native lora_name combo inputs.
"""
    SEARCH_ALIASES = ["lora", "lora name", "load lora"]


class EnviralColorMatchMethod(_NameSelectorBase):
    INPUT_NAME = "method"
    DISPLAY_LABEL = "Color match method"
    OPTIONS_PROVIDER = staticmethod(_color_match_methods)
    RETURN_TYPES = (_ExactComboOutput(_color_match_methods),)
    RETURN_NAMES = ("method",)
    OUTPUT_TOOLTIPS = ("Native-compatible ColorMatch method combo.",)
    DESCRIPTION = """
Selects or adapts a color match method for legacy ColorMatch method combo inputs.
"""
    SEARCH_ALIASES = ["color match", "color match method", "method"]


class EnviralColorMatchV2Method(_NameSelectorBase):
    INPUT_NAME = "method"
    DISPLAY_LABEL = "Color match V2 method"
    OPTIONS_PROVIDER = staticmethod(_color_match_v2_methods)
    RETURN_TYPES = (_ExactComboOutput(_color_match_v2_methods),)
    RETURN_NAMES = ("method",)
    OUTPUT_TOOLTIPS = ("Native-compatible ColorMatch V2 method combo.",)
    DESCRIPTION = """
Selects or adapts a color match method for V2 method combo inputs.
"""
    SEARCH_ALIASES = [
        "color match",
        "color match method",
        "color match v2",
        "method",
        "reinhard_lab_gpu",
    ]


NODE_CLASS_MAPPINGS = {
    "EnviralSamplerName": EnviralSamplerName,
    "EnviralSchedulerName": EnviralSchedulerName,
    "EnviralCheckpointName": EnviralCheckpointName,
    "EnviralVAEName": EnviralVAEName,
    "EnviralLoraName": EnviralLoraName,
    "EnviralColorMatchMethod": EnviralColorMatchMethod,
    "EnviralColorMatchV2Method": EnviralColorMatchV2Method,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnviralSamplerName": "Enviral Sampler Name",
    "EnviralSchedulerName": "Enviral Scheduler Name",
    "EnviralCheckpointName": "Enviral Checkpoint Name",
    "EnviralVAEName": "Enviral VAE Name",
    "EnviralLoraName": "Enviral LoRA Name",
    "EnviralColorMatchMethod": "Enviral Color Match Method",
    "EnviralColorMatchV2Method": "Enviral Color Match V2 Method",
}
