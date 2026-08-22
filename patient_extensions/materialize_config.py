#!/usr/bin/env python
"""Materialize machine-local HUP065/HUP080 configs from public templates.

The committed templates preserve executed scientific values while replacing
machine-specific paths and authorization text.  Materialized files are local,
ignored by Git, hash-locked to the current checkout, and receive a new hash;
they are never represented as the byte-identical executed freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
NON_SOURCE_SUFFIXES = {
    ".edf",
    ".joblib",
    ".npy",
    ".npz",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".zip",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def write_json(path: Path, payload: Any, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def expand(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        result = value
        for key, replacement in variables.items():
            result = result.replace("${" + key + "}", replacement)
        return str(Path(result)) if value.startswith("${") else result
    if isinstance(value, list):
        return [expand(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: expand(item, variables) for key, item in value.items()}
    return value


def _json_pointer(parent: str, key: str | int) -> str:
    encoded = str(key).replace("~", "~0").replace("/", "~1")
    return f"{parent}/{encoded}"


def _checkout_file(path_value: Any) -> Path | None:
    if not isinstance(path_value, str):
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        resolved = path.resolve()
        resolved.relative_to(REPO_ROOT.resolve())
    except (OSError, ValueError):
        return None
    if not resolved.is_file() or resolved.suffix.casefold() in NON_SOURCE_SUFFIXES:
        return None
    return resolved


def _line_ending_identity(path: Path, expected: str, observed: str) -> str:
    if expected == observed:
        return "byte_identical"
    payload = path.read_bytes()
    lf = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    variants = {hashlib.sha256(lf).hexdigest()}
    variants.add(hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest())
    if expected in variants and observed in variants:
        return "eol_only"
    return "relocation_specific_generated_or_public_adaptation"


def refresh_checkout_locks(value: Any, pointer: str = "") -> list[dict[str, str]]:
    """Refresh every source/reference lock that resolves inside this checkout.

    Executed byte hashes remain in the committed templates. This is deliberately
    limited to files below ``REPO_ROOT`` and excludes serialized models/raw
    archives, so no private science artifact becomes a public runtime dependency.
    """

    records: list[dict[str, str]] = []
    seen: set[str] = set()

    def update(path_value: Any, expected_value: Any, set_hash: Any, location: str) -> None:
        if not isinstance(expected_value, str) or not HEX64.fullmatch(expected_value):
            return
        path = _checkout_file(path_value)
        if path is None or location in seen:
            return
        expected = expected_value.casefold()
        observed = sha256_file(path)
        set_hash(observed)
        seen.add(location)
        records.append(
            {
                "json_pointer": location,
                "public_path": path.relative_to(REPO_ROOT.resolve()).as_posix(),
                "executed_or_template_sha256": expected,
                "checkout_sha256": observed,
                "identity": _line_ending_identity(path, expected, observed),
            }
        )

    def walk(node: Any, location: str) -> None:
        if isinstance(node, list):
            if len(node) == 2:
                update(
                    node[0],
                    node[1],
                    lambda digest: node.__setitem__(1, digest),
                    _json_pointer(location, 1),
                )
            for index, item in enumerate(node):
                walk(item, _json_pointer(location, index))
            return
        if not isinstance(node, dict):
            return

        if "path" in node and "sha256" in node:
            update(
                node["path"],
                node["sha256"],
                lambda digest: node.__setitem__("sha256", digest),
                _json_pointer(location, "sha256"),
            )
        for hash_key in tuple(node):
            if not hash_key.endswith("_sha256"):
                continue
            base = hash_key[: -len("_sha256")]
            path_key = next(
                (candidate for candidate in (base, base + "_path") if candidate in node),
                None,
            )
            if path_key is not None:
                update(
                    node[path_key],
                    node[hash_key],
                    lambda digest, key=hash_key: node.__setitem__(key, digest),
                    _json_pointer(location, hash_key),
                )
        for key, item in node.items():
            walk(item, _json_pointer(location, key))

    walk(value, pointer)
    records.sort(key=lambda row: row["json_pointer"])
    return records


def attach_relocation_receipt(payload: dict[str, Any], records: list[dict[str, str]]) -> None:
    portability = payload.setdefault("public_portability", {})
    portability["checkout_lock_policy"] = (
        "all source/reference locks resolving inside REPO_ROOT use the current "
        "checkout bytes; executed/template hashes remain provenance"
    )
    portability["checkout_lock_relocations"] = records


def unresolved_tokens(value: Any) -> Iterable[str]:
    if isinstance(value, str) and "${" in value:
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from unresolved_tokens(item)
    elif isinstance(value, list):
        for item in value:
            yield from unresolved_tokens(item)


def materialize_shared(variables: dict[str, str], *, force: bool) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    jobs = (
        (
            HERE / "common" / "protocol" / "protocol.template.json",
            HERE / "common" / "protocol" / "protocol.json",
            "protocol",
        ),
        (
            HERE / "common" / "legacy_adapters" / "stage_contract.template.json",
            HERE / "common" / "legacy_adapters" / "stage_contract.json",
            "stage_contract",
        ),
    )
    for template, output, key in jobs:
        payload = expand(load_json(template), variables)
        authority = payload.get("authority_freeze")
        if isinstance(authority, dict):
            executed_names = list(authority.get("allowed_new_output_directory_names", []))
            for public_name in ("HUP065", "HUP080"):
                if public_name not in authority["allowed_new_output_directory_names"]:
                    authority["allowed_new_output_directory_names"].append(public_name)
            payload.setdefault("public_portability", {})[
                "executed_allowed_new_output_directory_names"
            ] = executed_names
        canonical_locks = payload.get("canonical_hup060_locks")
        if isinstance(canonical_locks, dict):
            payload.setdefault("public_portability", {})[
                "executed_canonical_hup060_lock_sha256"
            ] = {
                name: str(lock["sha256"]).casefold()
                for name, lock in canonical_locks.items()
                if isinstance(lock, dict) and "sha256" in lock
            }
        records = refresh_checkout_locks(payload)
        attach_relocation_receipt(payload, records)
        write_json(output, payload, force=force)
        outputs[key] = output
    return outputs


def materialize_hup065(variables: dict[str, str], *, force: bool) -> Path:
    root = HERE / "HUP065"
    audit = expand(load_json(root / "SOURCE_ADAPTATION_AUDIT.template.json"), variables)
    protocol_path = HERE / "common" / "protocol" / "protocol.json"
    audit["shared_protocol_sha256"] = sha256_file(protocol_path)
    for relative in list(audit["scientific_implementation_sources"]):
        source = root / relative
        if source.is_file():
            audit["scientific_implementation_sources"][relative] = sha256_file(source)
    audit_records = refresh_checkout_locks(audit)
    attach_relocation_receipt(audit, audit_records)
    audit_path = root / "SOURCE_ADAPTATION_AUDIT.json"
    write_json(audit_path, audit, force=force)

    config = expand(load_json(root / "config.template.json"), variables)
    shared = config["shared_protocol"]
    shared["sha256"] = sha256_file(protocol_path)
    shared["common_module_sha256"] = sha256_file(Path(shared["common_module_path"]))
    executed_canonical_locks = {
        name: str(pair[1]).casefold()
        for name, pair in config["canonical"]["locks"].items()
    }
    config["implementation_audit"]["sha256"] = sha256_file(audit_path)
    config["public_portability"]["materialized_from_template_sha256"] = sha256_file(
        root / "config.template.json"
    )
    config["public_portability"]["path_variables"] = sorted(variables)
    config["public_portability"][
        "executed_canonical_lock_sha256"
    ] = executed_canonical_locks
    config_records = refresh_checkout_locks(config)
    attach_relocation_receipt(config, config_records)
    leftovers = list(unresolved_tokens(config))
    if leftovers:
        raise ValueError(f"unresolved template variables: {leftovers}")
    output = root / "config.json"
    write_json(output, config, force=force)
    return output


def materialize_hup080(variables: dict[str, str], *, force: bool) -> Path:
    root = HERE / "HUP080"
    audit = expand(load_json(root / "SOURCE_BINDING_AUDIT.template.json"), variables)
    protocol_path = HERE / "common" / "protocol" / "protocol.json"
    audit["shared_protocol_sha256"] = sha256_file(protocol_path)
    formal_adapter = audit.get("formal_adapter", {})
    if Path(str(formal_adapter.get("path", ""))).is_file():
        formal_adapter["sha256"] = sha256_file(Path(formal_adapter["path"]))
    audit_records = refresh_checkout_locks(audit)
    attach_relocation_receipt(audit, audit_records)
    audit_path = root / "SOURCE_BINDING_AUDIT.json"
    write_json(audit_path, audit, force=force)

    adapter = expand(
        load_json(root / "renderer_adapter" / "adapter_config.template.json"), variables
    )
    adapter_records = refresh_checkout_locks(adapter)
    attach_relocation_receipt(adapter, adapter_records)
    adapter_path = root / "renderer_adapter" / "adapter_config.json"
    write_json(adapter_path, adapter, force=force)

    config = expand(load_json(root / "config.template.json"), variables)
    v1_receipt = Path(str(config["exploratory_v2_delta"]["v1_failure_public_receipt"]))
    if not v1_receipt.is_file():
        raise FileNotFoundError(f"bundled parent-v1 provenance receipt is absent: {v1_receipt}")
    config["exploratory_v2_delta"]["v1_failure_public_receipt_sha256"] = sha256_file(
        v1_receipt
    )
    shared = config["shared_protocol"]
    shared["sha256"] = sha256_file(protocol_path)
    shared["common_protocol_py_sha256"] = sha256_file(Path(shared["common_protocol_py"]))
    shared["static_self_test_py_sha256"] = sha256_file(Path(shared["static_self_test_py"]))
    shared["readme_sha256"] = sha256_file(Path(shared["readme"]))
    split = Path(str(config["source_data"]["split_proposal"]))
    if split.is_file():
        config["source_data"]["split_proposal_sha256"] = sha256_file(split)
    config["public_portability"]["materialized_from_template_sha256"] = sha256_file(
        root / "config.template.json"
    )
    config["public_portability"]["path_variables"] = sorted(variables)
    config_records = refresh_checkout_locks(config)
    attach_relocation_receipt(config, config_records)
    leftovers = list(unresolved_tokens(config))
    if leftovers:
        raise ValueError(f"unresolved template variables: {leftovers}")
    output = root / "config.json"
    write_json(output, config, force=force)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", required=True, choices=("HUP065", "HUP080"))
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--legacy-workspace-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--legacy-extension-root",
        type=Path,
        default=HERE,
        help="optional public output/provenance parent; never a parent-v1 science input",
    )
    parser.add_argument(
        "--private-provenance-root",
        type=Path,
        default=HERE / "common" / "legacy_adapters",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    patient_root = HERE / args.patient
    variables = {
        "REPO_ROOT": str(REPO_ROOT.resolve()),
        "DS004100_ROOT": str(args.dataset_root.resolve()),
        "PATIENT_RUN_ROOT": str(patient_root.resolve()),
        "LEGACY_WORKSPACE_ROOT": str(args.legacy_workspace_root.resolve()),
        "LEGACY_EXTENSION_ROOT": str(args.legacy_extension_root.resolve()),
        "PRIVATE_PROVENANCE_ROOT": str(args.private_provenance_root.resolve()),
    }
    shared = materialize_shared(variables, force=args.force)
    output = (
        materialize_hup065(variables, force=args.force)
        if args.patient == "HUP065"
        else materialize_hup080(variables, force=args.force)
    )
    result = {
        "status": "materialized",
        "patient": args.patient,
        "config": str(output),
        "config_sha256": sha256_file(output),
        "executed_config_sha256": load_json(output)["public_portability"].get(
            "executed_config_sha256"
        )
        or load_json(output)["public_portability"].get("executed_v2_config_sha256"),
        "byte_identical_to_executed_freeze": False,
        "shared_outputs": {key: str(path) for key, path in shared.items()},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
