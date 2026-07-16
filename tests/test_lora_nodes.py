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


if __name__ == "__main__":
    unittest.main()
