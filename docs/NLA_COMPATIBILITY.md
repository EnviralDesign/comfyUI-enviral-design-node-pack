# NLA Compatibility

This repo is positioned as a stable companion utility/node pack for NLA AI
Video Creator-oriented ComfyUI workflows. It is not documented here as a hard
runtime dependency of NLA AI Video Creator.

If a specific NLA workflow package requires this pack, that workflow package
should say so directly and list the exact nodes it uses.

## What Currently Works

The current node pack exposes small utility nodes that are useful in
NLA-oriented ComfyUI workflows and in standalone ComfyUI workflows:

- `Text Split (Delimiter)` for fixed-output prompt or metadata splitting.
- `Lazy Index Switch` for lazy selection between up to 20 optional inputs.
- `Enviral Image Resize Kit` for pass-through, explicit resize, inside fit, and
  outside fit image preparation with optional mask handling.
- `Enviral Color Match V2` for reference-based color transfer using
  `color-matcher` CPU methods or the ComfyUI/Kornia Lab-space Reinhard path.
- `Model Patch Torch Settings` for narrowly scoped PyTorch FP16 accumulation
  callback patching on a ComfyUI `MODEL`.
- `WAN Resolution Snap` for preset or custom divisor-safe dimensions.
- `LM Studio Unified (URL + API Key)` for OpenAI-compatible local or remote LM
  Studio chat calls with optional image input.

## Planned

The following workflow packages are planned but not currently bundled:

- image generation provider workflow
- image-to-video provider workflow
- resize, snap, and color-match helper workflows

See [`../workflows/README.md`](../workflows/README.md) for the current workflow
package status.

## Stability Requirements for NLA-Oriented Workflows

To remain a reliable companion for NLA-oriented workflows, these surfaces should
remain stable:

- node class mapping keys in `NODE_CLASS_MAPPINGS`
- display names in `NODE_DISPLAY_NAME_MAPPINGS`
- input socket names, types, defaults, and required/optional status
- output socket order, types, and names
- resize mode names and aliases used by linked string or combo nodes
- `WAN Resolution Snap` preset names, snap modes, divisor behavior, and summary
  format
- `Text Split (Delimiter)` output order, including `part_1` through `part_20`,
  `remainder`, and `count`
- `Lazy Index Switch` input names `input_0` through `input_19` and lazy
  validation behavior
- `LM Studio Unified (URL + API Key)` URL normalization and API-key fallback
  behavior

Breaking any of these should be treated as a saved-workflow compatibility risk.

## Non-Goals

- Do not make this repo a large provider workflow bundle.
- Do not add broad all-in-one orchestration nodes.
- Do not imply NLA AI Video Creator requires this repo unless that dependency is
  present in the NLA project or in a specific NLA workflow package.
