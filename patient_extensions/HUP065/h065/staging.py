from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import shutil
from typing import Any, Callable, Iterator, Mapping

from .contracts import assert_path_within, iter_files, sha256_file, sha256_json


SUCCESS = "_SUCCESS.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = base64.urlsafe_b64encode(secrets.token_bytes(16)).decode("ascii").rstrip("=")
    temporary = path.with_name(f".j-{token}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def artifact_manifest(directory: Path) -> list[dict[str, Any]]:
    base = Path(directory).resolve()
    rows: list[dict[str, Any]] = []
    for path in iter_files(base):
        relative = path.relative_to(base).as_posix()
        if relative in {"artifact_manifest.json", SUCCESS}:
            continue
        rows.append(
            {"relative_path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return rows


def validate_completed_phase(directory: Path) -> dict[str, Any]:
    root = Path(directory)
    success_path = root / SUCCESS
    manifest_path = root / "artifact_manifest.json"
    if not success_path.is_file() or not manifest_path.is_file():
        raise FileExistsError(f"existing phase is incomplete and cannot be resumed: {root}")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if success.get("manifest_sha256") != sha256_json(rows):
        raise PermissionError(f"phase manifest receipt changed: {root}")
    for row in rows:
        path = root / str(row["relative_path"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise PermissionError(f"phase artifact missing or resized: {path}")
        if sha256_file(path) != str(row["sha256"]):
            raise PermissionError(f"phase artifact hash changed: {path}")
    return success


class PhaseStore:
    """Atomic, no-overwrite phase publication with crash-visible staging."""

    def __init__(self, science_root: Path, config_sha256: str) -> None:
        self.root = Path(science_root).resolve()
        self.artifacts = self.root / "artifacts"
        self.staging = self.root / ".staging"
        self.config_sha256 = str(config_sha256)

    def path(self, phase: str) -> Path:
        return assert_path_within(self.artifacts / phase, self.root)

    def require(self, phase: str) -> Path:
        path = self.path(phase)
        validate_completed_phase(path)
        return path

    @contextmanager
    def publish(self, phase: str, dependency: str | None) -> Iterator[Path | None]:
        final = self.path(phase)
        if final.exists():
            validate_completed_phase(final)
            yield None
            return
        if dependency is not None:
            self.require(dependency)
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=True, exist_ok=True)
        stage = assert_path_within(self.staging / phase, self.root)
        if stage.exists():
            raise FileExistsError(
                f"crash-visible staging exists; do not retry blindly: {stage}"
            )
        stage.mkdir(parents=False, exist_ok=False)
        try:
            yield stage
            rows = artifact_manifest(stage)
            atomic_json(stage / "artifact_manifest.json", rows)
            success = {
                "schema_version": "atomic-phase-success-v1",
                "phase": phase,
                "completed_utc": utc_now(),
                "config_sha256": self.config_sha256,
                "manifest_sha256": sha256_json(rows),
                "artifact_count": len(rows),
            }
            atomic_json(stage / SUCCESS, success)
            if final.exists():
                raise FileExistsError(f"phase target appeared during staging: {final}")
            os.replace(stage, final)
        except BaseException:
            # Deliberately retain staging.  For run-03 this is an access receipt;
            # for every other phase it prevents an unaudited blind rerun.
            raise

    def initialize_root(self) -> None:
        if self.root.exists():
            children = list(self.root.iterdir())
            allowed = {"artifacts", ".staging", "run_identity.json"}
            unexpected = [path for path in children if path.name not in allowed]
            if unexpected:
                raise FileExistsError(f"science_root contains unmanaged entries: {unexpected}")
        else:
            self.root.mkdir(parents=False, exist_ok=False)
        identity = self.root / "run_identity.json"
        payload = {
            "schema_version": "science-run-identity-v1",
            "config_sha256": self.config_sha256,
            "no_overwrite": True,
        }
        if identity.exists():
            observed = json.loads(identity.read_text(encoding="utf-8"))
            if observed != payload:
                raise PermissionError("science root belongs to a different configuration")
        else:
            atomic_json(identity, payload)


def recursive_inventory(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root_index, root in enumerate(roots):
        base = Path(root).resolve()
        if not base.is_dir():
            raise FileNotFoundError(base)
        for path in iter_files(base):
            rows.append(
                {
                    "root_index": root_index,
                    "root": str(base),
                    "relative_path": path.relative_to(base).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def compare_inventories(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> None:
    if before != after:
        before_map = {(r["root_index"], r["relative_path"]): r for r in before}
        after_map = {(r["root_index"], r["relative_path"]): r for r in after}
        keys = sorted(set(before_map) | set(after_map))
        changed = [key for key in keys if before_map.get(key) != after_map.get(key)]
        raise PermissionError(f"HUP060 authority inventory drifted: {changed[:20]}")
