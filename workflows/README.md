# Workflow Packages

This directory tracks planned ComfyUI workflow packages for the Enviral Design
node pack. The packages below are not currently bundled as executable workflow
JSON files.

The goal is to keep examples small and provider-oriented: enough to show how
the nodes connect, without turning this repo into a large workflow distribution.

## Planned Packages

### Image generation provider workflow

Status: planned, not present.

Intended scope:

- show how provider image-generation outputs can feed helper nodes in this pack
- demonstrate `Text Split (Delimiter)` and `Lazy Index Switch` for small prompt
  or provider-selection routing
- include a before/after example when a provider workflow benefits from the
  helper nodes

### Image-to-video provider workflow

Status: planned, not present.

Intended scope:

- show how image outputs can be prepared for image-to-video provider workflows
- demonstrate `WAN Resolution Snap` for divisor-safe video dimensions
- demonstrate `Enviral Image Resize Kit` for pass-through, exact resize,
  inside-fit, and outside-fit preparation

### Resize, snap, and color-match helper workflows

Status: planned, not present.

Intended scope:

- provide focused helper workflows for resizing, padding, cropping, and
  resolution snapping
- show `Enviral Color Match V2` before/after usage with a reference image
- keep examples independent from NLA AI Video Creator so they remain useful for
  standalone ComfyUI users

## Compatibility Notes

When workflow packages are added, they should list their required custom nodes
and any provider-specific assumptions directly in the workflow README. Missing
workflows should remain marked as planned until the actual workflow files are
committed.
