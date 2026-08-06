import importlib.util
import sys
import types
import unittest
from pathlib import Path


LORA_NODES_PATH = Path(__file__).parents[1] / "lora_nodes.py"


def load_lora_nodes(lora_names):
    folder_paths = types.SimpleNamespace(
        get_filename_list=lambda folder_name: list(lora_names),
        get_full_path_or_raise=lambda folder_name, filename: filename,
    )
    sys.modules["folder_paths"] = folder_paths
    spec = importlib.util.spec_from_file_location("enviral_lora_nodes_test", LORA_NODES_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LoraNodeTests(unittest.TestCase):
    def test_lora_folder_prefixes_include_nested_folders(self):
        module = load_lora_nodes(
            [
                "root.safetensors",
                "sdxl/style.safetensors",
                "wan\\characters\\hero.safetensors",
                "wan/motion/walk.safetensors",
            ]
        )

        self.assertEqual(
            module._lora_folder_prefixes(module._EnviralLoraLoaderBase._lora_names()),
            ["All LoRAs", "sdxl", "wan", "wan/characters", "wan/motion"],
        )

    def test_lora_folder_match_has_a_path_boundary(self):
        module = load_lora_nodes([])

        self.assertTrue(module._lora_is_in_folder("wan\\motion\\walk.safetensors", "WAN"))
        self.assertFalse(module._lora_is_in_folder("wan2/model.safetensors", "wan"))
        self.assertFalse(module._lora_is_in_folder("sdxl/wan_style.safetensors", "wan"))
        self.assertTrue(module._lora_is_in_folder("anything.safetensors", "All LoRAs"))

    def test_filtered_loader_validation_rejects_lora_outside_folder(self):
        module = load_lora_nodes(
            ["sdxl/style.safetensors", "wan/motion/walk.safetensors"]
        )

        self.assertIs(
            module.EnviralLoadLoraFiltered.VALIDATE_INPUTS(
                "wan", "wan/motion/walk.safetensors"
            ),
            True,
        )
        self.assertEqual(
            module.EnviralLoadLoraFiltered.VALIDATE_INPUTS(
                "wan", "sdxl/style.safetensors"
            ),
            "LoRA 'sdxl/style.safetensors' is not inside folder 'wan'",
        )

    def test_filtered_loader_resolves_folder_separator_and_case(self):
        module = load_lora_nodes(["WAN\\Characters\\hero.safetensors"])

        self.assertEqual(
            module.EnviralLoadLoraFiltered._resolve_folder("wan/characters"),
            "WAN/Characters",
        )

    def test_filtered_loader_exposes_up_to_five_banks(self):
        module = load_lora_nodes(["wan/one.safetensors"])

        input_types = module.EnviralLoadLoraFiltered.INPUT_TYPES()
        self.assertEqual(
            list(input_types["required"]),
            [
                "model",
                "clip",
                "folder",
                "lora_name",
                "strength_model",
                "strength_clip",
                "bank_count",
            ],
        )
        self.assertEqual(input_types["required"]["bank_count"][1]["default"], 1)
        self.assertEqual(
            input_types["required"]["bank_count"][1]["max"],
            module.MAX_LORA_BANKS,
        )
        self.assertIn("lora_name_5", input_types["optional"])
        self.assertTrue(input_types["optional"]["allow_list"][1]["forceInput"])
        self.assertEqual(
            input_types["optional"]["strength_model_2"][1]["default"], 0.0
        )
        self.assertEqual(
            input_types["optional"]["strength_clip_5"][1]["default"], 0.0
        )

    def test_filtered_loader_bypasses_zero_strength_banks(self):
        module = load_lora_nodes(["wan/one.safetensors"])
        model = object()
        clip = object()
        loader = module.EnviralLoadLoraFiltered()

        self.assertIs(
            module.EnviralLoadLoraFiltered.VALIDATE_INPUTS(
                "All LoRAs",
                "",
                strength_model=0,
                strength_clip=0,
                bank_count=3,
                lora_name_2="not/a/lora.safetensors",
                strength_model_2=0,
                strength_clip_2=0,
            ),
            True,
        )
        self.assertEqual(
            loader.load_lora_filtered(
                model,
                clip,
                "All LoRAs",
                "",
                0,
                0,
                bank_count=3,
                lora_name_2="not/a/lora.safetensors",
                strength_model_2=0,
                strength_clip_2=0,
            ),
            (model, clip, "wan/one.safetensors"),
        )

    def test_allow_list_matches_display_names_and_full_paths(self):
        module = load_lora_nodes(
            [
                "wan/one.safetensors",
                "wan/nested/two.safetensors",
                "sdxl/other.safetensors",
            ]
        )

        self.assertEqual(
            module.EnviralLoadLoraFiltered._available_loras(
                "wan",
                "one.safetensors\nWAN/NESTED/TWO.SAFETENSORS\nmissing.safetensors",
            ),
            [
                ("wan/one.safetensors", "one.safetensors"),
                ("wan/nested/two.safetensors", "nested/two.safetensors"),
            ],
        )

    def test_filtered_loader_outputs_visible_lora_list(self):
        module = load_lora_nodes(
            [
                "wan/one.safetensors",
                "wan/nested/two.safetensors",
                "sdxl/other.safetensors",
            ]
        )
        loader = module.EnviralLoadLoraFiltered()

        self.assertEqual(
            loader.load_lora_filtered(
                "model",
                "clip",
                "wan",
                "",
                0,
                0,
                allow_list="nested/two.safetensors\none.safetensors",
            ),
            (
                "model",
                "clip",
                "one.safetensors\nnested/two.safetensors",
            ),
        )

    def test_allow_list_rejects_active_bank_outside_list(self):
        module = load_lora_nodes(["wan/one.safetensors", "wan/two.safetensors"])

        self.assertEqual(
            module.EnviralLoadLoraFiltered.VALIDATE_INPUTS(
                "wan",
                "wan/two.safetensors",
                allow_list="one.safetensors",
            ),
            "LoRA 'wan/two.safetensors' is not included by the current allow-list",
        )

    def test_filtered_loader_is_the_only_non_deprecated_lora_loader(self):
        module = load_lora_nodes([])

        self.assertTrue(module.EnviralLoadLora.DEPRECATED)
        self.assertFalse(module.EnviralLoadLoraFiltered.DEPRECATED)
        self.assertTrue(module.EnviralLoadLoraModelOnly.DEPRECATED)

    def test_filtered_loader_applies_active_banks_in_order(self):
        module = load_lora_nodes(
            ["wan/one.safetensors", "wan/three.safetensors", "sdxl/other.safetensors"]
        )
        loader = module.EnviralLoadLoraFiltered()
        calls = []

        def load_lora(model, clip, lora_name, strength_model, strength_clip):
            calls.append((model, clip, lora_name, strength_model, strength_clip))
            return (f"{model}:{lora_name}", f"{clip}:{lora_name}")

        loader.load_lora = load_lora
        result = loader.load_lora_filtered(
            "model",
            "clip",
            "wan",
            "wan/one.safetensors",
            1.0,
            0.0,
            bank_count=3,
            lora_name_2="sdxl/other.safetensors",
            strength_model_2=0.0,
            strength_clip_2=0.0,
            lora_name_3="wan/three.safetensors",
            strength_model_3=0.0,
            strength_clip_3=0.75,
        )

        self.assertEqual(
            calls,
            [
                ("model", "clip", "wan/one.safetensors", 1.0, 0.0),
                (
                    "model:wan/one.safetensors",
                    "clip:wan/one.safetensors",
                    "wan/three.safetensors",
                    0.0,
                    0.75,
                ),
            ],
        )
        self.assertEqual(
            result,
            (
                "model:wan/one.safetensors:wan/three.safetensors",
                "clip:wan/one.safetensors:wan/three.safetensors",
                "one.safetensors\nthree.safetensors",
            ),
        )


if __name__ == "__main__":
    unittest.main()
