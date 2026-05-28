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
