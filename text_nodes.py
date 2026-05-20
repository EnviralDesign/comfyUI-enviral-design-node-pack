DELIMITER_ESCAPES = (
    ("\\r\\n", "\r\n"),
    ("\\n", "\n"),
    ("\\r", "\r"),
    ("\\t", "\t"),
)


def _decode_delimiter(delimiter):
    value = str(delimiter or "")
    for source, target in DELIMITER_ESCAPES:
        value = value.replace(source, target)
    return value


class TextSplitByDelimiter:
    MAX_OUTPUTS = 8

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Input text to split."
                }),
                "delimiter": ("STRING", {
                    "default": ",",
                    "tooltip": "Delimiter used to split the input text. Supports \\n, \\r\\n, \\r, and \\t escapes."
                }),
                "output_count": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": cls.MAX_OUTPUTS,
                    "step": 1,
                    "tooltip": "How many leading parts to expose before the remainder bucket."
                }),
                "strip_parts": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Trim whitespace around each split part."
                }),
                "skip_empty": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Drop empty parts after splitting."
                }),
            }
        }

    RETURN_TYPES = ("STRING",) * MAX_OUTPUTS + ("STRING", "INT")
    RETURN_NAMES = (
        "part_1",
        "part_2",
        "part_3",
        "part_4",
        "part_5",
        "part_6",
        "part_7",
        "part_8",
        "remainder",
        "count",
    )
    FUNCTION = "split_text"
    CATEGORY = "EnviralDesign/text"

    def split_text(self, text, delimiter, output_count, strip_parts, skip_empty):
        delimiter = _decode_delimiter(delimiter)

        if delimiter == "":
            parts = [text]
            join_delimiter = ""
        else:
            parts = text.split(delimiter)
            join_delimiter = delimiter

        if strip_parts:
            parts = [part.strip() for part in parts]

        if skip_empty:
            parts = [part for part in parts if part != ""]

        count = len(parts)
        output_count = max(1, min(int(output_count), self.MAX_OUTPUTS))

        visible_parts = list(parts[:output_count])
        remainder_parts = parts[output_count:]

        while len(visible_parts) < self.MAX_OUTPUTS:
            visible_parts.append("")

        remainder = join_delimiter.join(remainder_parts) if remainder_parts else ""

        return (*visible_parts, remainder, count)


NODE_CLASS_MAPPINGS = {
    "EnviralTextSplitByDelimiter": TextSplitByDelimiter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnviralTextSplitByDelimiter": "Text Split (Delimiter)",
}
