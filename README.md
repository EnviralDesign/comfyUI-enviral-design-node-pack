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
- optional `image` input for vision-capable models

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
