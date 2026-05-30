# comfyUI-enviral-design-node-pack

Small, dependency-light ComfyUI utility nodes from Enviral Design.

The goal of this pack is to keep genuinely useful glue nodes in one place
without pulling in a large third-party custom-node bundle just to get one tiny
piece of functionality.

## Included

### `Text Split (Delimiter)`

Splits an input string using a delimiter and returns:

- `part_1` through `part_20`
- `remainder`
- `count`

Behavior:

- `output_count` controls how many leading parts are emitted before overflow is
  collected into `remainder`
- `strip_parts` trims whitespace around each part
- `skip_empty` removes empty parts after splitting
- `delimiter` supports common escape strings such as `\n`, `\r\n`, `\r`, and
  `\t`

This node intentionally uses a fixed socket layout instead of dynamic outputs so
saved workflows stay predictable and easy to debug.

### `Lazy Index Switch`

Selects one of `input_0` through `input_19` using a zero-based `index` value.

Behavior:

- accepts any ComfyUI type on its input sockets
- evaluates only the selected input branch
- pairs with combo-style nodes that expose an `INDEX` output
- reports a clear validation error if the selected input is not connected

### `Enviral Image Resize Kit`

Resizes an image with a small set of explicit workflow-friendly modes:

- `pass-through`: keeps the input image resolution and ignores width/height
- `explicit resize`: stretches the image to the requested width and height
- `inside fit`: preserves the full image, pads to the requested aspect ratio,
  then outputs the requested width and height
- `outside fit`: fills the requested aspect ratio and center-crops overflow

The mode input is a dropdown, but it also accepts linked strings such as
`resize`, `pad`, `crop`, and `pass-through`. This makes it pair cleanly with
combo nodes that output strings. Optional masks are transformed with the same
geometry and the node returns `image`, `width`, `height`, and `mask`.

### `Enviral Color Match V2`

Transfers color from a reference image batch to a target image batch.

Inputs:

- `image_target`: image batch to recolor
- `image_ref`: image batch to sample color from
- `method`: `mkl`, `hm`, `reinhard`, `mvgd`, `hm-mvgd-hm`,
  `hm-mkl-hm`, or `reinhard_lab_gpu`
- `strength`: blend/extrapolation amount, where `0` is unchanged and `1` is
  the matched result
- `multithread`: enables threaded CPU processing for multi-image batches

The CPU methods use `color-matcher`. The `reinhard_lab_gpu` method uses
ComfyUI's Kornia dependency for Lab-space matching.

### `Model Patch Torch Settings`

Patches a `MODEL` with ComfyUI model callbacks for PyTorch backend settings.

Inputs:

- `model`: model to clone and patch
- `enable_fp16_accumulation`: toggles
  `torch.backends.cuda.matmul.allow_fp16_accumulation` while the model runs

When enabled, the patched model turns full FP16 accumulation on before model
execution and turns it back off during cleanup. When disabled, it forces the
setting off before model execution.

### `LM Studio Unified (URL + API Key)`

Sends text, an image, or both to an LM Studio OpenAI-compatible chat endpoint.

Inputs:

- `base_url`: accepts `http://127.0.0.1:1234/v1`,
  `https://lmstudio.example.com`, or a full `/v1/chat/completions` URL
- `api_key`: optional bearer token for LM Studio API-token auth
- `api_key_env_var`: optional environment variable fallback when `api_key` is
  blank, defaulting to `LMSTUDIO_API_KEY`. On Windows, this checks the current
  process first, then the User and Machine environment variable registry values.
- `model`, `prompt`, `system_prompt`, `seed`, `max_tokens`, `temperature`, and
  `timeout_seconds`
- optional `image` input for vision-capable models. If the input is an IMAGE
  batch, each frame is sent as a separate `image_url` content part, capped by
  `max_images`

For a secured Cloudflare route, use a public `base_url` such as:

```text
https://lmstudio.enviral-design.com
```

The node will call:

```text
https://lmstudio.enviral-design.com/v1/chat/completions
```

## Install

Clone this repo into `ComfyUI/custom_nodes` and restart ComfyUI.

```text
ComfyUI/custom_nodes/comfyUI-enviral-design-node-pack
```

## Donations & Support

If this saves you time, you can support the work here:

- [Patreon](https://www.patreon.com/EnviralDesign)
- [GitHub Sponsors](https://github.com/sponsors/EnviralDesign)
- [PayPal](https://www.paypal.com/donate?hosted_button_id=RP8EJAHSDTZ86)
