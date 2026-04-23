# comfyUI-enviral-design-node-pack

Small, dependency-light ComfyUI utility nodes from Enviral Design.

The goal of this pack is to keep genuinely useful glue nodes in one place
without pulling in a large third-party custom-node bundle just to get one tiny
piece of functionality.

## Included

### `Text Split (Delimiter)`

Splits an input string using a delimiter and returns:

- `part_1` through `part_8`
- `remainder`
- `count`

Behavior:

- `output_count` controls how many leading parts are emitted before overflow is
  collected into `remainder`
- `strip_parts` trims whitespace around each part
- `skip_empty` removes empty parts after splitting

This node intentionally uses a fixed socket layout instead of dynamic outputs so
saved workflows stay predictable and easy to debug.

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
