# Shared protocol snapshot

`protocol.template.json` is a path-relocated public template derived from the
executed shared HUP060-aligned protocol. Scientific parameters, split roles,
selection rules, gate thresholds, seeds, and training schedules are retained.
Only machine-local paths are placeholders.

`common_protocol.py` is the reviewed shared protocol implementation. Run
`materialize_config.py` from the parent directory before a local rerun; this
creates an ignored `protocol.json` and refreshes only path-bound file hashes.
The resulting file is a relocation-specific configuration, not the original
freeze.

