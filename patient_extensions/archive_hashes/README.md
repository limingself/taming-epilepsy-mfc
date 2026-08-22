# Private archive hash anchors

The complete byte-level archives are intentionally not mirrored to GitHub.
Their per-file SHA-256 manifests provide immutable provenance anchors:

| Subject | Private archive role | SHA-256 of `FULL_ARCHIVE_MANIFEST.json` |
|---|---|---|
| HUP065 | final sparse-control archive v1 | `df6752e1916eb12e1bb0d16330fc77ce63e4444846d675cddcc4213c3f36a637` |
| HUP080 | final exploratory partial-actuation archive v2 | `6382eb5aec69643855866765ac39eb981f93abb6b99727e145ddb7d803ad4f0d` |

These hashes identify the private manifests; they do not grant access to raw
EEG, signal arrays, serialized models, training histories, or one-time access
receipts. The portable templates in this directory are not the original
hash-frozen configurations. Their relationship to the executed files is
recorded in `../PORTABILITY_MAP.json`.
