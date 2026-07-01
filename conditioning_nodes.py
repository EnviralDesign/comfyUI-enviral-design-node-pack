import logging

import node_helpers
import torch


LOGGER = logging.getLogger(__name__)
SOURCE_VERSION = "2.2.0"
MAX_TORCH_SEED = 0xFFFFFFFFFFFFFFFF


class EnviralKrea2SeedVarianceEnhancer:
    """Add seeded noise to text conditioning embeddings for Krea2/Z-Image style workflows.

    Adapted from ChangeTheConstants/SeedVarianceEnhancer v2.2.0, released under
    the MIT No Attribution License.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "randomize_percent": (
                    "FLOAT",
                    {
                        "default": 50.0,
                        "min": 1.0,
                        "max": 100.0,
                        "step": 1,
                        "tooltip": "Percentage of embedding values that receive random noise.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "min": -0xFFFFFFFF,
                        "max": 0xFFFFFFFF,
                        "step": 0.00001,
                        "tooltip": "Scale of the random noise. Add 1 billion to use the v2.1 seed behavior.",
                    },
                ),
                "noise_insert": (
                    [
                        "noise on beginning steps",
                        "noise on ending steps",
                        "noise on all steps",
                        "disabled",
                    ],
                    {
                        "tooltip": "Which generation steps use the noisy text embedding.",
                    },
                ),
                "steps_switchover_percent": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "min": 1.0,
                        "max": 99.0,
                        "step": 1,
                        "tooltip": "Percentage of sampler steps before switching between noisy and original embeddings.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_TORCH_SEED,
                        "control_after_generate": True,
                        "tooltip": "Seed used for embedding-value selection and noise generation.",
                    },
                ),
                "mask_starts_at": (
                    ["beginning", "end"],
                    {
                        "tooltip": "Which end of the prompt is protected from noise first.",
                    },
                ),
                "mask_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 99.0,
                        "step": 1,
                        "tooltip": "Percentage of the prompt protected from noise.",
                    },
                ),
                "log_to_console": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": f"Log tensor statistics and suggested strength values. Source version {SOURCE_VERSION}.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "randomize_conditioning"
    CATEGORY = "EnviralDesign/conditioning"
    DESCRIPTION = """
Seeded conditioning noise enhancer adapted for Krea2/Z-Image style workflows.
"""
    SEARCH_ALIASES = [
        "krea2",
        "seed variance",
        "seed variance enhancer",
        "conditioning noise",
        "z image",
    ]

    @staticmethod
    def _clamp_percent(value, minimum, maximum):
        return max(minimum, min(maximum, float(value))) / 100.0

    @staticmethod
    def _copy_conditioning_item(item):
        return [item[0], item[1].copy()]

    @staticmethod
    def _make_generator(tensor, seed):
        seed = int(seed) & MAX_TORCH_SEED
        try:
            generator = torch.Generator(device=tensor.device)
        except Exception:
            generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def log_tensor_statistics(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            LOGGER.warning("Seed variance enhancer received conditioning with no tensor.")
            return

        first_null, last_nonnull, null_sequences = self.tensor_first_null_sequence(tensor)

        if last_nonnull < tensor.size(1) - 1:
            sliced_tensor = tensor[:, :last_nonnull + 1, :]
            mean = torch.mean(sliced_tensor).item()
            std = torch.std(sliced_tensor).item()
            min_val = torch.min(sliced_tensor).item()
            max_val = torch.max(sliced_tensor).item()
        else:
            mean = torch.mean(tensor).item()
            std = torch.std(tensor).item()
            min_val = torch.min(tensor).item()
            max_val = torch.max(tensor).item()

        LOGGER.info("Embedding tensor statistics from seed variance enhancer:")
        if first_null != -1:
            number_of_null_seq = sum(1 for item in null_sequences if item == 0)
            LOGGER.info(
                "Null sequences: first=%s last_nonnull=%s total=%s",
                first_null,
                last_nonnull,
                number_of_null_seq,
            )
        LOGGER.info(
            "Dimensions: %s Min: %.6f Max: %.6f Mean: %.6f Standard deviation: %.6f Try strength in range %.6f - %.6f",
            ", ".join(map(str, tensor.shape)),
            min_val,
            max_val,
            mean,
            std,
            std / 10,
            std * 10,
        )

    @staticmethod
    def tensor_first_null_sequence(tensor):
        first_null = -1
        last_nonnull = -1
        null_sequences = [0] * tensor.size(1)

        if tensor.dim() == 3:
            for index in range(tensor.size(1)):
                sequence = tensor[:, index, ...]
                is_all_zero = torch.all(sequence == 0)
                null_sequences[index] = 0 if is_all_zero else 1

                if not is_all_zero:
                    last_nonnull = index
                if is_all_zero and first_null == -1:
                    first_null = index

        return first_null, last_nonnull, null_sequences

    def _select_conditioning_pair(self, conditioning, noise_insert):
        if len(conditioning) == 1:
            selected = self._copy_conditioning_item(conditioning[0])
            return selected, self._copy_conditioning_item(conditioning[0])

        if noise_insert == "noise on beginning steps":
            return (
                self._copy_conditioning_item(conditioning[0]),
                self._copy_conditioning_item(conditioning[1]),
            )
        if noise_insert == "noise on ending steps":
            return (
                self._copy_conditioning_item(conditioning[1]),
                self._copy_conditioning_item(conditioning[0]),
            )

        first_tag = conditioning[0][1].get("SVH_tag")
        second_tag = conditioning[1][1].get("SVH_tag")
        if first_tag != "noisy" and second_tag == "noisy":
            selected = self._copy_conditioning_item(conditioning[1])
            return selected, self._copy_conditioning_item(conditioning[1])

        selected = self._copy_conditioning_item(conditioning[0])
        return selected, self._copy_conditioning_item(conditioning[0])

    def randomize_conditioning(
        self,
        conditioning,
        randomize_percent,
        strength,
        noise_insert,
        steps_switchover_percent,
        seed,
        mask_starts_at,
        mask_percent,
        log_to_console,
    ):
        steps_switchover_percent = self._clamp_percent(steps_switchover_percent, 1, 99)
        randomize_percent = self._clamp_percent(randomize_percent, 1, 100)
        mask_percent = self._clamp_percent(mask_percent, 0, 99)

        if (
            len(conditioning) < 1
            or len(conditioning[0]) < 2
            or (len(conditioning) >= 2 and len(conditioning[1]) < 2)
        ):
            if log_to_console:
                LOGGER.warning("Seed variance enhancer received empty conditioning. Passing it through unchanged.")
            return (conditioning,)

        if strength == 0:
            if log_to_console:
                LOGGER.warning("Seed variance enhancer is disabled because strength is zero.")
                self.log_tensor_statistics(conditioning[0][0])
            return (conditioning,)

        if noise_insert == "disabled":
            if log_to_console:
                LOGGER.warning("Seed variance enhancer is disabled.")
                self.log_tensor_statistics(conditioning[0][0])
            return (conditioning,)

        if len(conditioning) > 2 and log_to_console:
            LOGGER.warning("Seed variance enhancer will only use the first two conditioning embeddings.")

        reset_seed = True
        if int(strength / 1000000000) == 1:
            strength -= 1000000000
            reset_seed = False
            if log_to_console:
                LOGGER.info(
                    "Detected 1 billion added to strength; subtracting it and using v2.1 seed behavior."
                )

        selected, other = self._select_conditioning_pair(conditioning, noise_insert)
        embedding = selected[0]
        metadata = selected[1]

        if not isinstance(embedding, torch.Tensor):
            if log_to_console:
                LOGGER.warning("Seed variance enhancer received conditioning with no tensor. Passing it through unchanged.")
            return (conditioning,)

        if log_to_console:
            self.log_tensor_statistics(embedding)

        noise_generator = self._make_generator(embedding, seed)
        noise = (
            torch.rand(
                embedding.shape,
                dtype=embedding.dtype,
                device=embedding.device,
                generator=noise_generator,
            )
            * 2
            * strength
            - strength
        )
        mask_generator = self._make_generator(embedding, seed + 1) if reset_seed else noise_generator
        noise_mask = torch.bernoulli(
            torch.full_like(embedding, randomize_percent),
            generator=mask_generator,
        ).bool()

        first_null, last_nonnull, null_sequences = self.tensor_first_null_sequence(embedding)

        if mask_percent > 0 or last_nonnull < embedding.size(1) - 1:
            if last_nonnull < embedding.size(1) - 1 and last_nonnull >= 0:
                seq_len = last_nonnull + 1
            else:
                seq_len = embedding.size(1)

            if mask_starts_at == "end":
                mask_start = seq_len - int(seq_len * mask_percent)
                mask_end = embedding.size(1)
            else:
                mask_start = 0
                mask_end = int(seq_len * mask_percent)

            prompt_mask = torch.arange(
                embedding.size(1),
                device=embedding.device,
            ).view(1, -1, 1).expand(embedding.size(0), -1, embedding.size(2))
            prompt_mask = (prompt_mask >= mask_start) & (prompt_mask < mask_end)

            if first_null > -1:
                if log_to_console:
                    LOGGER.info("Seed variance enhancer is masking null sequences from noise.")
                null_mask_tensor = ~torch.tensor(
                    null_sequences,
                    device=embedding.device,
                    dtype=torch.bool,
                )
                null_mask_tensor = null_mask_tensor.view(1, -1, 1).expand(
                    embedding.size(0),
                    -1,
                    embedding.size(2),
                )
                prompt_mask = prompt_mask | null_mask_tensor

            noise_mask = noise_mask & (~prompt_mask)

        noisy_tensor = embedding + (noise * noise_mask)
        noisy_embedding = [[noisy_tensor, metadata]]
        other_embedding = [other]

        if noise_insert == "noise on beginning steps":
            new_conditioning = node_helpers.conditioning_set_values(
                noisy_embedding,
                {
                    "start_percent": 0.0,
                    "end_percent": steps_switchover_percent,
                    "SVH_tag": "noisy",
                },
            )
            new_conditioning += node_helpers.conditioning_set_values(
                other_embedding,
                {
                    "start_percent": steps_switchover_percent,
                    "end_percent": 1.0,
                },
            )
            return (new_conditioning,)

        if noise_insert == "noise on ending steps":
            new_conditioning = node_helpers.conditioning_set_values(
                other_embedding,
                {
                    "start_percent": 0.0,
                    "end_percent": steps_switchover_percent,
                },
            )
            new_conditioning += node_helpers.conditioning_set_values(
                noisy_embedding,
                {
                    "start_percent": steps_switchover_percent,
                    "end_percent": 1.0,
                    "SVH_tag": "noisy",
                },
            )
            return (new_conditioning,)

        metadata.pop("start_percent", None)
        metadata.pop("end_percent", None)
        return (noisy_embedding,)


NODE_CLASS_MAPPINGS = {
    "EnviralKrea2SeedVarianceEnhancer": EnviralKrea2SeedVarianceEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnviralKrea2SeedVarianceEnhancer": "Enviral Krea2 Seed Variance Enhancer",
}
