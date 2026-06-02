# comfyUI-enviral-design-node-pack

Small, dependency-light ComfyUI utility nodes from Enviral Design.

The goal of this pack is to keep genuinely useful glue nodes in one place
without pulling in a large third-party custom-node bundle just to get one tiny
piece of functionality.

## Companion use with LatentSlate

This pack is intended to be a stable companion utility pack for
LatentSlate-oriented ComfyUI workflows while remaining useful on its own.

The nodes here are intentionally small, stable, and dependency-light. They cover
workflow glue that is useful around provider-driven image generation,
image-to-video, resizing, resolution snapping, color matching, local LLM calls,
and narrow PyTorch model-setting patches.

This repo does not claim to be a required dependency for LatentSlate.
If a LatentSlate workflow package needs one of these nodes, that workflow should
list the dependency explicitly. Planned workflow package notes live in
[`workflows/README.md`](workflows/README.md).

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
  fills padding with `letterbox_color`, then outputs the requested width and
  height
- `outside fit`: fills the requested aspect ratio and center-crops overflow

Inputs:

- `image`: input image batch
- `mode`: resize mode dropdown or linked string alias
- `width` and `height`: requested output dimensions, ignored by `pass-through`
- `letterbox_color`: used only by `inside fit`; accepts simple names, `R,G,B`,
  `0.0-1.0` float triples, or `#RRGGBB`
- optional `mask`: transformed with the same geometry as the image

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
ComfyUI's Kornia dependency for Lab-space matching and requires both inputs to
be RGB IMAGE tensors with 3 channels.

### `Enviral Load LoRA`

Native-style LoRA loader that keeps ComfyUI's LoRA application behavior but
allows `lora_name` to be driven by a linked string.

Inputs:

- `model`: diffusion model to patch
- `clip`: CLIP model to patch
- `lora_name`: dropdown or linked string matching a LoRA path known to ComfyUI
- `strength_model`: diffusion model LoRA strength
- `strength_clip`: CLIP LoRA strength

Outputs:

- patched `model`
- patched `clip`

### `Enviral Load LoRA (Model Only)`

Model-only variant for workflows such as WAN where the LoRA chain patches the
diffusion model without a CLIP connection.

Inputs:

- `model`: diffusion model to patch
- `lora_name`: dropdown or linked string matching a LoRA path known to ComfyUI
- `strength_model`: diffusion model LoRA strength

Output:

- patched `model`

### `Model Patch Torch Settings`

Experimental node that patches a compatible ComfyUI `MODEL` with callbacks for
one CUDA PyTorch backend setting.

Inputs:

- `model`: model to clone and patch
- `enable_fp16_accumulation`: toggles
  `torch.backends.cuda.matmul.allow_fp16_accumulation` while the model runs

When enabled, the patched model turns full FP16 accumulation on before model
execution and turns it back off during cleanup. When disabled, it forces the
setting off before model execution. This requires a PyTorch build that exposes
`torch.backends.cuda.matmul.allow_fp16_accumulation` and a ComfyUI model patcher
with `clone()` and `add_callback()` support.

### `WAN Resolution Snap`

Converts a preset or custom typed resolution into divisor-safe dimensions for
WAN video workflows.

Inputs:

- `preset`: `custom` or common square, landscape, portrait, and 4:3 presets
- `width` and `height`: freeform custom dimensions, used when `preset` is
  `custom`
- `snap`: `nearest`, `floor`, or `ceil`
- `divisible_by`: output divisor, defaulting to `32`
- `swap`: swaps width and height before snapping

Outputs:

- `width` and `height` as integers
- `width_float` and `height_float` for nodes that expect float sockets
- `summary`, such as `768 x 1344 | 1.03 MP | 4:7 | div32`

If snapping changes the requested dimensions, `summary` also appends the source
size, for example `| from 1000 x 540`.

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
- `user_agent`: HTTP User-Agent override for routes that block Python's default
  urllib signature
- `debug`: prints request details to the ComfyUI console

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

Most nodes use ComfyUI's existing Python, PyTorch, Pillow, and NumPy
environment. `Enviral Color Match V2` also uses `color-matcher` for its CPU
methods; that dependency is declared in `pyproject.toml`. If you install this
repo manually and see a missing `color_matcher` import, install
`color-matcher` into the same Python environment that runs ComfyUI.

## Workflow examples

This repo currently documents nodes and planned workflow packages. It does not
currently bundle executable workflow JSON files. See
[`workflows/README.md`](workflows/README.md) for the planned package list and
status.

## Donations & Support

If this saves you time, you can support the work here:

- [Patreon](https://www.patreon.com/EnviralDesign)
- [GitHub Sponsors](https://github.com/sponsors/EnviralDesign)
- [PayPal](https://www.paypal.com/donate?hosted_button_id=RP8EJAHSDTZ86)
