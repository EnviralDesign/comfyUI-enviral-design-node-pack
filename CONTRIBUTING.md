# Contributing

Thanks for helping keep this pack small and stable.

## Node Style Rules

- Keep sockets stable. Avoid renaming inputs, outputs, return names, node class
  keys, or display names after release unless there is a migration plan.
- Prefer fixed socket layouts for saved-workflow stability. Dynamic behavior is
  fine internally, but workflows should stay predictable when reopened.
- Keep dependencies minimal. Do not add large runtime dependencies for a small
  helper node unless there is no reasonable ComfyUI-native or standard-library
  alternative.
- Avoid giant all-in-one node bundles. Prefer small composable nodes with a
  narrow job and obvious inputs/outputs.
- Document every node in `README.md` or a focused doc under `docs/` before
  release.
- Provide before/after workflow examples when possible, especially for image
  resizing, snapping, color matching, and provider-compatibility helpers.
- Keep standalone ComfyUI use valid. NLA-oriented helpers should still make
  sense outside NLA AI Video Creator workflows.

## Compatibility Expectations

Changes that can break saved workflows need extra care:

- socket type changes
- required input additions
- output order changes
- node key or display name changes
- dependency additions
- behavior changes that alter image dimensions, masks, prompt text, or model
  patch timing

When a breaking change is unavoidable, document the change clearly and add a
replacement path or compatibility alias when practical.

## Pull Request Checklist

- The changed or added node is documented.
- New dependencies are justified and declared in `pyproject.toml`.
- Existing node socket layouts are unchanged, or the PR explains the migration.
- Relevant examples or workflow notes are updated.
- Basic checks pass, such as Python compilation or any future test suite.
