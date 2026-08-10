import importlib.util
import sys
import types
import unittest
from pathlib import Path


MODEL_LOADER_NODES_PATH = Path(__file__).parents[1] / "model_loader_nodes.py"


def load_model_loader_nodes(files_by_category):
    def get_full_path_or_raise(category, filename):
        if filename not in files_by_category.get(category, []):
            raise ValueError(f"Unknown {category} file: {filename}")
        return f"C:/models/{category}/{filename}"

    folder_paths = types.SimpleNamespace(
        get_filename_list=lambda category: list(files_by_category.get(category, [])),
        get_full_path_or_raise=get_full_path_or_raise,
        get_folder_paths=lambda category: [f"C:/models/{category}"],
    )
    sys.modules["folder_paths"] = folder_paths
    spec = importlib.util.spec_from_file_location(
        "enviral_model_loader_nodes_test", MODEL_LOADER_NODES_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelLoaderNodeTests(unittest.TestCase):
    def setUp(self):
        self.module = load_model_loader_nodes(
            {
                "checkpoints": [
                    "sdxl/base.safetensors",
                    "wan/v2/model.safetensors",
                    "root.safetensors",
                ],
                "diffusion_models": [
                    "ltx/v2/model.safetensors",
                    "wan/dit.safetensors",
                ],
                "text_encoders": [
                    "ltx/umt5.safetensors",
                    "wan/umt5.safetensors",
                ],
            }
        )

    def test_checkpoint_folders_include_nested_paths(self):
        loader = self.module.EnviralLoadCheckpointFiltered

        self.assertEqual(
            loader._folder_input()[1]["options"],
            ["All Checkpoints", "sdxl", "wan", "wan/v2"],
        )

    def test_checkpoint_filter_accepts_display_names_and_full_paths(self):
        loader = self.module.EnviralLoadCheckpointFiltered

        self.assertEqual(
            loader._available_files(
                "wan",
                "v2/model.safetensors\nSDXL/BASE.SAFETENSORS\nmissing.safetensors",
            ),
            [("wan/v2/model.safetensors", "v2/model.safetensors")],
        )

    def test_checkpoint_validation_enforces_folder_and_allow_list(self):
        loader = self.module.EnviralLoadCheckpointFiltered

        self.assertIs(
            loader.VALIDATE_INPUTS(
                "wan",
                ckpt_name="wan/v2/model.safetensors",
                allow_list="v2/model.safetensors",
            ),
            True,
        )
        self.assertEqual(
            loader.VALIDATE_INPUTS("wan", ckpt_name="sdxl/base.safetensors"),
            "ckpt_name 'sdxl/base.safetensors' is not inside folder 'wan'",
        )
        self.assertEqual(
            loader.VALIDATE_INPUTS(
                "wan",
                ckpt_name="wan/v2/model.safetensors",
                allow_list="other.safetensors",
            ),
            "ckpt_name 'wan/v2/model.safetensors' is not included by the current allow-list",
        )

    def test_model_loader_inputs_have_one_selection_without_banks(self):
        checkpoint_inputs = self.module.EnviralLoadCheckpointFiltered.INPUT_TYPES()
        diffusion_inputs = self.module.EnviralLoadDiffusionModelFiltered.INPUT_TYPES()
        text_encoder_inputs = self.module.EnviralLoadTextEncoderFiltered.INPUT_TYPES()

        self.assertEqual(list(checkpoint_inputs["required"]), ["folder", "ckpt_name"])
        self.assertEqual(list(diffusion_inputs["required"]), ["folder", "unet_name", "weight_dtype"])
        self.assertEqual(list(text_encoder_inputs["required"]), ["folder", "clip_name", "type"])
        self.assertNotIn("bank_count", checkpoint_inputs["required"])
        self.assertTrue(checkpoint_inputs["optional"]["allow_list"][1]["forceInput"])
        self.assertEqual(diffusion_inputs["required"]["weight_dtype"][0][0], "default")
        self.assertEqual(text_encoder_inputs["optional"]["device"][0], ["default", "cpu"])

    def test_checkpoint_loader_keeps_the_relative_path_for_loading(self):
        calls = []

        def load_checkpoint_guess_config(path, **kwargs):
            calls.append((path, kwargs))
            return "model", "clip", "vae"

        sys.modules["comfy"] = types.SimpleNamespace(
            sd=types.SimpleNamespace(load_checkpoint_guess_config=load_checkpoint_guess_config)
        )
        loader = self.module.EnviralLoadCheckpointFiltered()

        self.assertEqual(
            loader.load_checkpoint("wan", "wan/v2/model.safetensors"),
            ("model", "clip", "vae", "v2/model.safetensors"),
        )
        self.assertEqual(calls[0][0], "C:/models/checkpoints/wan/v2/model.safetensors")
        self.assertEqual(calls[0][1]["embedding_directory"], ["C:/models/embeddings"])

    def test_diffusion_model_loader_uses_native_weight_dtype_option(self):
        calls = []

        def load_diffusion_model(path, model_options):
            calls.append((path, model_options))
            return "model"

        sys.modules["comfy"] = types.SimpleNamespace(
            sd=types.SimpleNamespace(load_diffusion_model=load_diffusion_model)
        )
        loader = self.module.EnviralLoadDiffusionModelFiltered()

        self.assertEqual(
            loader.load_diffusion_model("ltx", "ltx/v2/model.safetensors", "default"),
            ("model", "v2/model.safetensors"),
        )
        self.assertEqual(calls, [("C:/models/diffusion_models/ltx/v2/model.safetensors", {})])

    def test_text_encoder_loader_uses_native_clip_type(self):
        calls = []
        clip_type = types.SimpleNamespace(STABLE_DIFFUSION="sd", WAN="wan")

        def load_clip(**kwargs):
            calls.append(kwargs)
            return "clip"

        sys.modules["comfy"] = types.SimpleNamespace(
            sd=types.SimpleNamespace(CLIPType=clip_type, load_clip=load_clip)
        )
        loader = self.module.EnviralLoadTextEncoderFiltered()

        self.assertEqual(
            loader.load_text_encoder("wan", "wan/umt5.safetensors", type="wan"),
            ("clip", "umt5.safetensors"),
        )
        self.assertEqual(calls[0]["ckpt_paths"], ["C:/models/text_encoders/wan/umt5.safetensors"])
        self.assertEqual(calls[0]["clip_type"], "wan")


if __name__ == "__main__":
    unittest.main()
