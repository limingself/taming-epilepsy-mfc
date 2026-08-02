"""BIDS-ZIP ingestion, quality control, preprocessing, and segment caching.

The source dataset is intentionally treated as immutable.  In particular, the
three known seizure-offset anomalies are represented by :class:`OffsetQC`
records and by separate ``raw_offset_s`` / ``effective_offset_s`` fields; the
``events.tsv`` data frame returned by :meth:`BIDSZipDataset.metadata` is never
edited in place.

EDF members are streamed from a ZIP archive into a private temporary directory
because MNE requires a seekable filesystem path.  Only the requested EDF is
ever extracted, and the temporary copy is removed before the loader returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd
from scipy.signal import butter, firwin, resample_poly, sosfilt, upfirdn


CACHE_SCHEMA_VERSION = 4
DEFAULT_SUBJECTS = (
    "HUP060",
    "HUP064",
    "HUP065",
    "HUP080",
    "HUP086",
    "HUP114",
    "HUP146",
    "HUP160",
)


def causal_polyphase_resample(
    values: np.ndarray,
    source_sfreq: float,
    target_sfreq: float,
    *,
    axis: int = -1,
    half_length_factor: int = 10,
) -> np.ndarray:
    """Rationally resample with a causal anti-alias FIR and no delay removal.

    ``scipy.signal.resample_poly`` centers its symmetric FIR and consequently
    lets an output sample depend on input samples on both sides of its nominal
    time.  Here the same rational upsample/FIR/downsample structure is executed
    by :func:`scipy.signal.upfirdn`, but the FIR starts at lag zero and its group
    delay is retained.  Output sample ``k`` therefore depends only on source
    samples available by output time ``k / target_sfreq``.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[axis] < 1:
        raise ValueError("values must contain at least one sample")
    source = float(source_sfreq)
    target = float(target_sfreq)
    if not np.isfinite(source) or not np.isfinite(target) or source <= 0 or target <= 0:
        raise ValueError("source_sfreq and target_sfreq must be finite and positive")
    if int(half_length_factor) < 1:
        raise ValueError("half_length_factor must be positive")
    ratio = Fraction(target / source).limit_denominator(10_000)
    up, down = int(ratio.numerator), int(ratio.denominator)
    if up == down == 1:
        return array.copy()
    maximum_rate = max(up, down)
    half_length = int(half_length_factor) * maximum_rate
    taps = firwin(
        2 * half_length + 1,
        1.0 / maximum_rate,
        window=("kaiser", 5.0),
    )
    # Upsampling inserts ``up - 1`` zeros, so scale the FIR to retain DC gain.
    taps = np.asarray(taps * up, dtype=np.float64)
    return np.asarray(
        upfirdn(taps, array, up=up, down=down, axis=axis),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class OffsetQC:
    """A non-destructive correction for a known invalid seizure offset."""

    subject: str
    run: str
    raw_offset_s: float
    effective_offset_s: float
    note: str


# These values were pre-specified after inspection of the source sidecars.  All
# three raw offsets are retained verbatim in RunMetadata.events.  The corrected
# values are used only in derived duration/QC fields.
KNOWN_OFFSET_QC: tuple[OffsetQC, ...] = (
    OffsetQC(
        "HUP060",
        "run-03",
        553.99609375,
        253.998,
        "Raw offset exceeds RecordingDuration (313.998 s) and the event "
        "sample/time pair is inconsistent; use 253.998 s for derived QC only.",
    ),
    OffsetQC(
        "HUP114",
        "run-01",
        402.99609375,
        176.998,
        "Raw offset exceeds RecordingDuration (236.998 s) and the event "
        "sample/time pair is inconsistent; use 176.998 s for derived QC only.",
    ),
    OffsetQC(
        "HUP114",
        "run-03",
        393.99609375,
        171.998,
        "Raw offset exceeds RecordingDuration (231.998 s) and the event "
        "sample/time pair is inconsistent; use 171.998 s for derived QC only.",
    ),
)


def _normalise_subject(subject: str) -> str:
    value = str(subject).strip()
    if value.lower().startswith("sub-"):
        value = value[4:]
    value = value.upper()
    if not value:
        raise ValueError("subject cannot be empty")
    return value


def _normalise_run(run: str | int) -> str:
    if isinstance(run, int):
        return f"run-{run:02d}"
    value = str(run).strip().lower()
    if value.startswith("run-"):
        suffix = value[4:]
    else:
        suffix = value
    if suffix.isdigit():
        return f"run-{int(suffix):02d}"
    raise ValueError(f"invalid BIDS run label: {run!r}")


def _parse_bids_entities(filename: str) -> dict[str, str]:
    entities: dict[str, str] = {}
    for token in Path(filename).name.split("_"):
        if "-" not in token:
            continue
        key, value = token.split("-", 1)
        if key and value:
            entities[key] = value.split(".", 1)[0]
    return entities


def _read_text(archive: zipfile.ZipFile, member: str) -> str:
    with archive.open(member, "r") as stream:
        return stream.read().decode("utf-8-sig")


def _read_tsv(archive: zipfile.ZipFile, member: str | None) -> pd.DataFrame:
    if member is None:
        return pd.DataFrame()
    # BIDS uses literal n/a as a meaningful missing-data label.  Keeping it as
    # text is important for status_description and avoids losing SOZ labels.
    return pd.read_csv(
        StringIO(_read_text(archive, member)),
        sep="\t",
        keep_default_na=False,
        na_values=[""],
    )


def _read_json(archive: zipfile.ZipFile, member: str | None) -> dict[str, Any]:
    if member is None:
        return {}
    return json.loads(_read_text(archive, member))


def _first_numeric(frame: pd.DataFrame, mask: np.ndarray, column: str) -> float | None:
    if column not in frame.columns or not np.any(mask):
        return None
    value = pd.to_numeric(frame.loc[mask, column], errors="coerce").dropna()
    return None if value.empty else float(value.iloc[0])


@dataclass(frozen=True)
class BIDSRecord:
    """Archive members belonging to one BIDS iEEG recording."""

    subject: str
    session: str
    task: str
    acquisition: str
    run: str
    zip_path: Path
    edf_member: str
    channels_member: str | None
    events_member: str | None
    ieeg_json_member: str | None
    coordsystem_member: str | None
    electrodes_member: str | None
    edf_size_bytes: int
    edf_compressed_bytes: int
    edf_crc: int

    @property
    def key(self) -> str:
        return f"{self.subject}_{self.task}_{self.run}"

    @property
    def bids_basename(self) -> str:
        return Path(self.edf_member).name.removesuffix("_ieeg.edf")


@dataclass
class RunMetadata:
    """Parsed BIDS sidecars for one recording."""

    record: BIDSRecord
    channels: pd.DataFrame
    events: pd.DataFrame
    ieeg: dict[str, Any]
    coordsystem: dict[str, Any]
    electrodes: pd.DataFrame
    offset_qc: OffsetQC | None = None
    qc_notes: tuple[str, ...] = ()

    @property
    def good_channels(self) -> tuple[str, ...]:
        if "name" not in self.channels.columns:
            return ()
        if "status" not in self.channels.columns:
            return tuple(self.channels["name"].astype(str))
        status = self.channels["status"].astype(str).str.strip().str.casefold()
        return tuple(self.channels.loc[status.eq("good"), "name"].astype(str))

    @property
    def sampling_frequency(self) -> float | None:
        value = self.ieeg.get("SamplingFrequency")
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            pass
        if "sampling_frequency" in self.channels.columns:
            values = pd.to_numeric(
                self.channels["sampling_frequency"], errors="coerce"
            ).dropna()
            if not values.empty:
                return float(values.iloc[0])
        return None

    @property
    def recording_duration_s(self) -> float | None:
        try:
            value = self.ieeg.get("RecordingDuration")
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    def _event_mask(self, kind: str) -> np.ndarray:
        if self.events.empty:
            return np.zeros(0, dtype=bool)
        if "trial_type" in self.events.columns:
            labels = (
                self.events["trial_type"]
                .astype(str)
                .str.strip()
                .str.casefold()
                .str.replace("_", " ", regex=False)
            )
            mask = labels.str.contains(kind, regex=False).to_numpy()
            if np.any(mask):
                return mask
        if "value" in self.events.columns:
            values = pd.to_numeric(self.events["value"], errors="coerce")
            code = 2 if kind == "onset" else 1
            return values.eq(code).to_numpy()
        return np.zeros(len(self.events), dtype=bool)

    @property
    def onset_s(self) -> float | None:
        return _first_numeric(self.events, self._event_mask("onset"), "onset")

    @property
    def raw_offset_s(self) -> float | None:
        return _first_numeric(self.events, self._event_mask("offset"), "onset")

    @property
    def effective_offset_s(self) -> float | None:
        if self.offset_qc is not None:
            return float(self.offset_qc.effective_offset_s)
        return self.raw_offset_s

    @property
    def seizure_duration_s(self) -> float | None:
        onset, offset = self.onset_s, self.effective_offset_s
        if onset is None or offset is None:
            return None
        return float(offset - onset)

    def channel_descriptions(self, channel_names: Sequence[str]) -> tuple[str, ...]:
        if "name" not in self.channels.columns:
            return tuple("n/a" for _ in channel_names)
        description_column = (
            self.channels["status_description"]
            if "status_description" in self.channels.columns
            else pd.Series("n/a", index=self.channels.index)
        )
        mapping = dict(zip(self.channels["name"].astype(str), description_column.astype(str)))
        return tuple(mapping.get(name, "n/a") for name in channel_names)

    def coordinates_for(
        self, channel_names: Sequence[str]
    ) -> tuple[np.ndarray, str | None]:
        coordinates = np.full((len(channel_names), 3), np.nan, dtype=float)
        required = {"name", "x", "y", "z"}
        if required.issubset(self.electrodes.columns):
            table = self.electrodes.copy()
            table["name"] = table["name"].astype(str)
            table = table.drop_duplicates("name").set_index("name")
            for row, name in enumerate(channel_names):
                if name not in table.index:
                    continue
                values = pd.to_numeric(table.loc[name, ["x", "y", "z"]], errors="coerce")
                coordinates[row] = values.to_numpy(dtype=float)
        units = self.coordsystem.get("iEEGCoordinateUnits")
        return coordinates, None if units is None else str(units)


class BIDSZipDataset:
    """Read BIDS-style subject ZIPs without unpacking the dataset tree."""

    def __init__(
        self,
        zip_root: str | os.PathLike[str],
        subjects: Sequence[str] = DEFAULT_SUBJECTS,
        offset_qc_overrides_s: Mapping[str, float] | None = None,
    ) -> None:
        self.zip_root = Path(zip_root)
        self.subjects = tuple(_normalise_subject(value) for value in subjects)
        known = {(item.subject, item.run): item for item in KNOWN_OFFSET_QC}
        for key, corrected in (offset_qc_overrides_s or {}).items():
            match = re.fullmatch(r"(?:sub-)?(HUP\d+)_?(run-\d+)", str(key), re.I)
            if match is None:
                raise ValueError(f"invalid offset override key: {key!r}")
            subject = _normalise_subject(match.group(1))
            run = _normalise_run(match.group(2))
            previous = known.get((subject, run))
            raw = np.nan if previous is None else previous.raw_offset_s
            note = (
                "Configuration-supplied offset correction; raw events.tsv is retained."
                if previous is None
                else previous.note
            )
            known[(subject, run)] = OffsetQC(subject, run, raw, float(corrected), note)
        self._offset_qc = known
        self._record_cache: dict[str, tuple[BIDSRecord, ...]] = {}
        self._metadata_cache: dict[str, RunMetadata] = {}

    def zip_path(self, subject: str) -> Path:
        subject = _normalise_subject(subject)
        path = self.zip_root / f"sub-{subject}.zip"
        if not path.is_file():
            raise FileNotFoundError(f"BIDS subject archive not found: {path}")
        return path

    def records(self, subject: str, task: str | None = None) -> tuple[BIDSRecord, ...]:
        subject = _normalise_subject(subject)
        if subject not in self._record_cache:
            self._record_cache[subject] = self._scan_records(subject)
        records = self._record_cache[subject]
        if task is not None:
            wanted = str(task).strip().casefold()
            records = tuple(record for record in records if record.task.casefold() == wanted)
        return records

    def _scan_records(self, subject: str) -> tuple[BIDSRecord, ...]:
        zip_path = self.zip_path(subject)
        records: list[BIDSRecord] = []
        with zipfile.ZipFile(zip_path, "r") as archive:
            names = {info.filename: info for info in archive.infolist() if not info.is_dir()}
            for member, info in names.items():
                if not member.lower().endswith("_ieeg.edf"):
                    continue
                entities = _parse_bids_entities(member)
                if entities.get("sub", "").upper() != subject:
                    continue
                base = member[: -len("_ieeg.edf")]
                acquisition = entities.get("acq", "unknown")
                directory = member.rsplit("/", 1)[0] if "/" in member else ""

                def exact(suffix: str) -> str | None:
                    candidate = f"{base}_{suffix}"
                    return candidate if candidate in names else None

                coordsystems = [
                    name
                    for name in names
                    if name.startswith(f"{directory}/")
                    and f"_acq-{acquisition}_" in Path(name).name
                    and name.endswith("_coordsystem.json")
                ]
                electrodes = [
                    name
                    for name in names
                    if name.startswith(f"{directory}/")
                    and f"_acq-{acquisition}_" in Path(name).name
                    and name.endswith("_electrodes.tsv")
                ]
                records.append(
                    BIDSRecord(
                        subject=subject,
                        session=entities.get("ses", "unknown"),
                        task=entities.get("task", "unknown"),
                        acquisition=acquisition,
                        run=_normalise_run(entities.get("run", "1")),
                        zip_path=zip_path,
                        edf_member=member,
                        channels_member=exact("channels.tsv"),
                        events_member=exact("events.tsv"),
                        ieeg_json_member=exact("ieeg.json"),
                        coordsystem_member=sorted(coordsystems)[0] if coordsystems else None,
                        electrodes_member=sorted(electrodes)[0] if electrodes else None,
                        edf_size_bytes=int(info.file_size),
                        edf_compressed_bytes=int(info.compress_size),
                        edf_crc=int(info.CRC),
                    )
                )
        return tuple(sorted(records, key=lambda item: (item.task, item.run, item.acquisition)))

    def get_record(
        self,
        subject: str,
        task: str,
        run: str | int,
        acquisition: str | None = None,
    ) -> BIDSRecord:
        run = _normalise_run(run)
        matches = [
            record
            for record in self.records(subject, task=task)
            if record.run == run
            and (acquisition is None or record.acquisition == acquisition)
        ]
        if len(matches) != 1:
            raise KeyError(
                f"expected one record for {subject}/{task}/{run}, found {len(matches)}"
            )
        return matches[0]

    def metadata(self, record: BIDSRecord) -> RunMetadata:
        if record.key in self._metadata_cache:
            return self._metadata_cache[record.key]
        notes: list[str] = []
        with zipfile.ZipFile(record.zip_path, "r") as archive:
            channels = _read_tsv(archive, record.channels_member)
            events = _read_tsv(archive, record.events_member)
            ieeg = _read_json(archive, record.ieeg_json_member)
            coordsystem = _read_json(archive, record.coordsystem_member)
            electrodes = _read_tsv(archive, record.electrodes_member)
        if channels.empty:
            notes.append("missing channels.tsv")
        if record.task == "ictal" and events.empty:
            notes.append("missing ictal events.tsv")
        if not ieeg:
            notes.append("missing run-level ieeg.json; sampling rate falls back to channels.tsv")
        if electrodes.empty:
            notes.append("electrode coordinates unavailable in archive")
        offset_qc = self._offset_qc.get((record.subject, record.run)) if record.task == "ictal" else None
        metadata = RunMetadata(
            record=record,
            channels=channels,
            events=events,
            ieeg=ieeg,
            coordsystem=coordsystem,
            electrodes=electrodes,
            offset_qc=offset_qc,
            qc_notes=tuple(notes),
        )
        if offset_qc is not None:
            raw = metadata.raw_offset_s
            if raw is None:
                notes.append("known offset QC record exists but no raw offset was parsed")
            elif np.isfinite(offset_qc.raw_offset_s) and not np.isclose(
                raw, offset_qc.raw_offset_s, atol=1e-6
            ):
                notes.append(
                    f"known offset QC expected {offset_qc.raw_offset_s:g} s, observed {raw:g} s"
                )
            notes.append(offset_qc.note)
            metadata.qc_notes = tuple(notes)
        self._metadata_cache[record.key] = metadata
        return metadata

    def selected_records(
        self,
        subject: str,
        selections: Sequence[tuple[str, str | int]] | None = None,
    ) -> tuple[BIDSRecord, ...]:
        if selections is None:
            return self.records(subject)
        return tuple(self.get_record(subject, task, run) for task, run in selections)

    def common_good_channels(
        self,
        subject: str,
        records: Sequence[BIDSRecord] | None = None,
        selections: Sequence[tuple[str, str | int]] | None = None,
    ) -> tuple[str, ...]:
        """Return the ordered good-channel intersection across selected runs.

        Ordering always follows the first selected run, which gives every
        patient a deterministic channel basis without replacing electrode names
        by anonymous integer indices.
        """

        if records is not None and selections is not None:
            raise ValueError("pass records or selections, not both")
        chosen = tuple(
            records if records is not None else self.selected_records(subject, selections)
        )
        if not chosen:
            raise ValueError(f"no selected records for {subject}")
        good = [self.metadata(record).good_channels for record in chosen]
        common = set(good[0])
        for names in good[1:]:
            common.intersection_update(names)
        ordered = tuple(name for name in good[0] if name in common)
        if not ordered:
            raise ValueError(f"no common good channels across selected runs for {subject}")
        return ordered

    def build_manifest(self, subjects: Sequence[str] | None = None) -> pd.DataFrame:
        """Build a sidecar-only manifest; no EDF member is extracted."""

        rows: list[dict[str, Any]] = []
        for subject in subjects or self.subjects:
            zip_path = self.zip_path(subject)
            zip_size = zip_path.stat().st_size
            for record in self.records(subject):
                meta = self.metadata(record)
                status = (
                    meta.channels["status"].astype(str).str.strip().str.casefold()
                    if "status" in meta.channels.columns
                    else pd.Series(dtype=str)
                )
                onset_mask = meta._event_mask("onset")
                offset_mask = meta._event_mask("offset")
                onset_sample = _first_numeric(meta.events, onset_mask, "sample")
                offset_sample = _first_numeric(meta.events, offset_mask, "sample")
                sfreq = meta.sampling_frequency

                def sample_mismatch(sample: float | None, seconds: float | None) -> bool:
                    if sample is None or seconds is None or sfreq is None:
                        return False
                    return abs(sample / sfreq - seconds) > max(0.01, 1.5 / sfreq)

                raw_offset = meta.raw_offset_s
                duration = meta.recording_duration_s
                rows.append(
                    {
                        "subject": record.subject,
                        "session": record.session,
                        "task": record.task,
                        "run": record.run,
                        "acquisition": record.acquisition,
                        "zip_path": str(record.zip_path),
                        "edf_member": record.edf_member,
                        "zip_size_bytes": int(zip_size),
                        "edf_size_bytes": record.edf_size_bytes,
                        "edf_compressed_bytes": record.edf_compressed_bytes,
                        "has_channels_tsv": record.channels_member is not None,
                        "has_events_tsv": record.events_member is not None,
                        "has_ieeg_json": record.ieeg_json_member is not None,
                        "has_coordsystem_json": record.coordsystem_member is not None,
                        "has_electrodes_tsv": record.electrodes_member is not None,
                        "n_channels": int(len(meta.channels)),
                        "n_good_channels": int(status.eq("good").sum()),
                        "n_bad_channels": int(status.eq("bad").sum()),
                        "sampling_frequency_hz": sfreq,
                        "recording_duration_s": duration,
                        "onset_s": meta.onset_s,
                        "raw_offset_s": raw_offset,
                        "effective_offset_s": meta.effective_offset_s,
                        "seizure_duration_s": meta.seizure_duration_s,
                        "offset_qc_override": meta.offset_qc is not None,
                        "raw_offset_outside_recording": bool(
                            raw_offset is not None
                            and duration is not None
                            and raw_offset > duration + 1.0 / (sfreq or 1.0)
                        ),
                        "onset_sample_time_mismatch": sample_mismatch(
                            onset_sample, meta.onset_s
                        ),
                        "offset_sample_time_mismatch": sample_mismatch(
                            offset_sample, raw_offset
                        ),
                        "qc_notes": " | ".join(meta.qc_notes),
                    }
                )
        return pd.DataFrame(rows).sort_values(
            ["subject", "task", "run"], ignore_index=True
        )


def _description_tokens(description: str) -> set[str]:
    return {
        token.strip().casefold()
        for token in re.split(r"[,;|]", str(description))
        if token.strip()
    }


@dataclass
class ProcessedSegment:
    """A channel-by-time preprocessed EEG segment in microvolts."""

    data: np.ndarray
    channel_names: tuple[str, ...]
    status_descriptions: tuple[str, ...]
    coordinates: np.ndarray
    coordinate_units: str | None
    subject: str
    task: str
    run: str
    start_s: float
    duration_s: float
    source_sfreq: float
    sfreq: float
    units: str = "uV"
    qc_notes: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    cache_path: Path | None = None
    cache_hit: bool = False

    @property
    def ch_names(self) -> tuple[str, ...]:
        return self.channel_names

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.data.shape[1], dtype=float) / self.sfreq

    @property
    def soz_mask(self) -> np.ndarray:
        return np.array(
            ["soz" in _description_tokens(value) for value in self.status_descriptions],
            dtype=bool,
        )

    @property
    def resect_mask(self) -> np.ndarray:
        return np.array(
            ["resect" in _description_tokens(value) for value in self.status_descriptions],
            dtype=bool,
        )

    def to_npz(self, path: str | os.PathLike[str]) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "subject": self.subject,
            "task": self.task,
            "run": self.run,
            "start_s": self.start_s,
            "duration_s": self.duration_s,
            "source_sfreq": self.source_sfreq,
            "sfreq": self.sfreq,
            "units": self.units,
            "coordinate_units": self.coordinate_units,
            "qc_notes": list(self.qc_notes),
            "provenance": self.provenance,
        }
        handle = tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}-",
            suffix=".npz",
            dir=destination.parent,
            delete=False,
        )
        temporary = Path(handle.name)
        handle.close()
        try:
            np.savez_compressed(
                temporary,
                data=np.asarray(self.data, dtype=np.float32),
                channel_names=np.asarray(self.channel_names, dtype=str),
                status_descriptions=np.asarray(self.status_descriptions, dtype=str),
                coordinates=np.asarray(self.coordinates, dtype=float),
                metadata_json=np.asarray(json.dumps(payload, ensure_ascii=False, sort_keys=True)),
            )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def from_npz(cls, path: str | os.PathLike[str]) -> "ProcessedSegment":
        source = Path(path)
        with np.load(source, allow_pickle=False) as archive:
            payload = json.loads(str(archive["metadata_json"].item()))
            return cls(
                data=np.asarray(archive["data"], dtype=np.float32),
                channel_names=tuple(str(value) for value in archive["channel_names"]),
                status_descriptions=tuple(
                    str(value) for value in archive["status_descriptions"]
                ),
                coordinates=np.asarray(archive["coordinates"], dtype=float),
                coordinate_units=payload["coordinate_units"],
                subject=payload["subject"],
                task=payload["task"],
                run=payload["run"],
                start_s=float(payload["start_s"]),
                duration_s=float(payload["duration_s"]),
                source_sfreq=float(payload["source_sfreq"]),
                sfreq=float(payload["sfreq"]),
                units=payload["units"],
                qc_notes=tuple(payload["qc_notes"]),
                provenance=dict(payload["provenance"]),
                cache_path=source,
                cache_hit=True,
            )


class EEGSegmentLoader:
    """Load, filter, reference, resample, and cache EEG crops.

    The forward-only Butterworth SOS bandpass is causal.  Set
    ``resampling_mode='causal_upfirdn'`` for prediction/control experiments
    that require strict forecast-boundary causality.  The legacy
    ``'symmetric_polyphase'`` mode remains available for backward-compatible
    descriptive analyses and requires a boundary guard.
    """

    def __init__(
        self,
        dataset: BIDSZipDataset,
        cache_dir: str | os.PathLike[str] | None = None,
        target_sfreq: float = 256.0,
        bandpass_hz: tuple[float, float] = (1.0, 50.0),
        filter_order: int = 4,
        burn_in_s: float = 20.0,
        reference: str = "car",
        temp_dir: str | os.PathLike[str] | None = None,
        resampling_mode: str = "symmetric_polyphase",
        causal_fir_half_length_factor: int = 10,
    ) -> None:
        self.dataset = dataset
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.target_sfreq = float(target_sfreq)
        self.bandpass_hz = tuple(float(value) for value in bandpass_hz)
        self.filter_order = int(filter_order)
        self.burn_in_s = float(burn_in_s)
        self.reference = str(reference).strip().casefold()
        self.resampling_mode = str(resampling_mode).strip().casefold()
        self.causal_fir_half_length_factor = int(causal_fir_half_length_factor)
        if temp_dir is not None:
            self.temp_dir = Path(temp_dir)
        elif self.cache_dir is not None:
            self.temp_dir = self.cache_dir / "_edf_tmp"
        else:
            self.temp_dir = Path.cwd() / "cache" / "_edf_tmp"
        self.mne_home_dir = self.temp_dir.parent / "_mne_home"
        low, high = self.bandpass_hz
        if not (0 < low < high):
            raise ValueError(f"invalid bandpass: {self.bandpass_hz}")
        if self.target_sfreq <= 2 * high:
            raise ValueError("target_sfreq must exceed twice the high cutoff")
        if self.filter_order < 1 or self.burn_in_s < 0:
            raise ValueError("filter_order must be positive and burn_in_s non-negative")
        if self.reference != "car":
            raise ValueError("only common-average reference ('car') is supported")
        if self.resampling_mode not in {"symmetric_polyphase", "causal_upfirdn"}:
            raise ValueError(
                "resampling_mode must be 'symmetric_polyphase' or 'causal_upfirdn'"
            )
        if self.causal_fir_half_length_factor < 1:
            raise ValueError("causal_fir_half_length_factor must be positive")

    def load_ictal(
        self,
        subject: str,
        run: str | int,
        duration_s: float = 20.0,
        channel_names: Sequence[str] | None = None,
        use_cache: bool = True,
    ) -> ProcessedSegment:
        record = self.dataset.get_record(subject, "ictal", run)
        metadata = self.dataset.metadata(record)
        if metadata.onset_s is None:
            raise ValueError(f"no seizure onset found for {record.key}")
        if metadata.effective_offset_s is not None and (
            metadata.onset_s + duration_s > metadata.effective_offset_s + 1e-9
        ):
            raise ValueError(
                f"requested {duration_s:g} s ictal crop exceeds the effective seizure "
                f"offset for {record.key}"
            )
        return self.load(
            record,
            start_s=metadata.onset_s,
            duration_s=duration_s,
            channel_names=channel_names,
            use_cache=use_cache,
        )

    def load_interictal(
        self,
        subject: str,
        run: str | int,
        start_s: float = 60.0,
        duration_s: float = 30.0,
        channel_names: Sequence[str] | None = None,
        use_cache: bool = True,
    ) -> ProcessedSegment:
        record = self.dataset.get_record(subject, "interictal", run)
        return self.load(
            record,
            start_s=start_s,
            duration_s=duration_s,
            channel_names=channel_names,
            use_cache=use_cache,
        )

    def load(
        self,
        record: BIDSRecord,
        start_s: float,
        duration_s: float,
        channel_names: Sequence[str] | None = None,
        use_cache: bool = True,
    ) -> ProcessedSegment:
        start_s, duration_s = float(start_s), float(duration_s)
        if start_s < 0 or duration_s <= 0:
            raise ValueError("start_s must be non-negative and duration_s positive")
        metadata = self.dataset.metadata(record)
        names = (
            metadata.good_channels
            if channel_names is None
            else tuple(str(name) for name in channel_names)
        )
        if not names:
            raise ValueError(f"no channels selected for {record.key}")
        if len(set(names)) != len(names):
            raise ValueError("selected channel names must be unique")
        not_good = [name for name in names if name not in metadata.good_channels]
        if not_good:
            raise ValueError(
                f"selected channels are absent or not status=good in {record.key}: {not_good}"
            )
        cache_path = self._cache_path(record, start_s, duration_s, names)
        if use_cache and cache_path is not None and cache_path.is_file():
            return ProcessedSegment.from_npz(cache_path)

        segment = self._read_and_preprocess(record, metadata, start_s, duration_s, names)
        if use_cache and cache_path is not None:
            segment.to_npz(cache_path)
            segment.cache_path = cache_path
        return segment

    def _cache_path(
        self,
        record: BIDSRecord,
        start_s: float,
        duration_s: float,
        channel_names: Sequence[str],
    ) -> Path | None:
        if self.cache_dir is None:
            return None
        stat = record.zip_path.stat()
        key = {
            "schema": CACHE_SCHEMA_VERSION,
            "zip_size": stat.st_size,
            "zip_mtime_ns": stat.st_mtime_ns,
            "edf_member": record.edf_member,
            "edf_crc": record.edf_crc,
            "edf_size": record.edf_size_bytes,
            "start_s": start_s,
            "duration_s": duration_s,
            "channels": list(channel_names),
            "target_sfreq": self.target_sfreq,
            "bandpass_hz": self.bandpass_hz,
            "filter_order": self.filter_order,
            "burn_in_s": self.burn_in_s,
            "reference": self.reference,
            "resampling_mode": self.resampling_mode,
            "causal_fir_half_length_factor": self.causal_fir_half_length_factor,
        }
        digest = sha256(
            json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        filename = (
            f"{record.bids_basename}_start-{start_s:.3f}s_dur-{duration_s:.3f}s_"
            f"fs-{self.target_sfreq:g}_{digest}.npz"
        )
        return self.cache_dir / record.subject / filename

    def _import_mne(self):
        # The managed desktop environment may expose a read-only user profile.
        # MNE reads its config at import time, so direct it to the writable
        # system temporary directory without altering the user's config.
        self.mne_home_dir.mkdir(parents=True, exist_ok=True)
        (self.mne_home_dir / ".mne").mkdir(parents=True, exist_ok=True)
        os.environ["_MNE_FAKE_HOME_DIR"] = str(self.mne_home_dir)
        os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
        import mne  # noqa: PLC0415

        return mne

    def _read_and_preprocess(
        self,
        record: BIDSRecord,
        metadata: RunMetadata,
        start_s: float,
        duration_s: float,
        channel_names: tuple[str, ...],
    ) -> ProcessedSegment:
        mne = self._import_mne()
        notes = list(metadata.qc_notes)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        edf_path = self.temp_dir / (
            f"{record.subject}-{record.run}-{record.edf_crc:08x}-"
            f"{os.getpid()}-{threading.get_ident()}.edf"
        )
        raw = None
        try:
            with zipfile.ZipFile(record.zip_path, "r") as archive:
                with archive.open(record.edf_member, "r") as source, edf_path.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
            raw = mne.io.read_raw_edf(
                edf_path,
                preload=False,
                infer_types=False,
                verbose="ERROR",
            )
            source_sfreq = float(raw.info["sfreq"])
            low, high = self.bandpass_hz
            if high >= source_sfreq / 2:
                raise ValueError(
                    f"high cutoff {high:g} Hz is not below EDF Nyquist "
                    f"({source_sfreq / 2:g} Hz) for {record.key}"
                )
            lookup: dict[str, int] = {}
            for index, raw_name in enumerate(raw.ch_names):
                normalised = raw_name.strip().casefold()
                if normalised in lookup:
                    raise ValueError(f"duplicate EDF channel label after normalisation: {raw_name}")
                lookup[normalised] = index
            missing = [name for name in channel_names if name.strip().casefold() not in lookup]
            if missing:
                raise ValueError(f"BIDS good channels missing from EDF {record.key}: {missing}")
            picks = [lookup[name.strip().casefold()] for name in channel_names]

            target_start_sample = int(round(start_s * source_sfreq))
            target_samples = int(round(duration_s * source_sfreq))
            burn_samples_requested = int(round(self.burn_in_s * source_sfreq))
            read_start = max(0, target_start_sample - burn_samples_requested)
            burn_samples = target_start_sample - read_start
            read_stop = target_start_sample + target_samples
            if read_stop > raw.n_times:
                available = max(0.0, raw.n_times / source_sfreq - start_s)
                raise ValueError(
                    f"requested crop exceeds EDF duration for {record.key}; "
                    f"only {available:.3f} s available"
                )
            if burn_samples < burn_samples_requested:
                notes.append(
                    f"filter burn-in shortened from {self.burn_in_s:g} s to "
                    f"{burn_samples / source_sfreq:g} s at recording boundary"
                )
            data_v = raw.get_data(picks=picks, start=read_start, stop=read_stop)
        finally:
            if raw is not None:
                raw.close()
            edf_path.unlink(missing_ok=True)

        if not np.isfinite(data_v).all():
            raise ValueError(f"non-finite EDF values in requested crop for {record.key}")
        # MNE expresses electrophysiology channels in SI volts.  Microvolts keep
        # the numerical range convenient while retaining an explicit unit.
        data_uv = np.asarray(data_v, dtype=np.float64) * 1e6
        sos = butter(
            self.filter_order,
            self.bandpass_hz,
            btype="bandpass",
            fs=source_sfreq,
            output="sos",
        )
        filtered_full = sosfilt(sos, data_uv, axis=-1)
        # Common-average reference is instantaneous and evaluated only over the
        # patient-wise good-channel basis supplied to this call.
        referenced_full = filtered_full - filtered_full.mean(axis=0, keepdims=True)
        ratio = Fraction(self.target_sfreq / source_sfreq).limit_denominator(10_000)
        expected = int(round(duration_s * self.target_sfreq))
        if self.resampling_mode == "causal_upfirdn":
            # Resample the complete burn-in plus requested interval.  Cropping
            # occurs on the output time grid and the causal FIR group delay is
            # intentionally retained; no future sample can enter an output.
            resampled_full = causal_polyphase_resample(
                referenced_full,
                source_sfreq,
                self.target_sfreq,
                axis=-1,
                half_length_factor=self.causal_fir_half_length_factor,
            )
            crop_start = int(
                np.ceil(burn_samples * ratio.numerator / ratio.denominator)
            )
            crop_stop = crop_start + expected
            if resampled_full.shape[1] < crop_stop:
                raise RuntimeError(
                    f"causal resampling produced {resampled_full.shape[1]} samples, "
                    f"but time crop requires {crop_stop}"
                )
            resampled = resampled_full[:, crop_start:crop_stop]
            maximum_rate = max(ratio.numerator, ratio.denominator)
            half_length = self.causal_fir_half_length_factor * maximum_rate
            causal_group_delay_s = half_length / (source_sfreq * ratio.numerator)
            guard_s = 0.0
            guard_samples = 0
            notes.append(
                "strictly causal FIR/upfirdn resampling; FIR group delay retained and "
                "forecast scoring starts at the true boundary without a guard"
            )
        else:
            referenced = referenced_full[:, burn_samples : burn_samples + target_samples]
            resampled = resample_poly(
                referenced, ratio.numerator, ratio.denominator, axis=-1
            )
            resampled = resampled[:, :expected]
            causal_group_delay_s = float("nan")
            guard_s = 0.125
            guard_samples = int(np.ceil(guard_s * self.target_sfreq))
            notes.append(
                "polyphase resampling uses a symmetric (non-causal) FIR; forecast "
                f"scoring requires a {guard_samples}-sample ({guard_s:g} s) boundary guard"
            )
        if resampled.shape[1] < expected:
            raise RuntimeError(
                f"resampling produced {resampled.shape[1]} samples, expected {expected}"
            )
        resampled = resampled[:, :expected]
        # Numerical resampling error can reintroduce a ~1e-15 common mode.
        resampled -= resampled.mean(axis=0, keepdims=True)

        descriptions = metadata.channel_descriptions(channel_names)
        coordinates, coordinate_units = metadata.coordinates_for(channel_names)
        provenance = {
            "zip_path": str(record.zip_path),
            "edf_member": record.edf_member,
            "edf_crc": record.edf_crc,
            "channels_member": record.channels_member,
            "events_member": record.events_member,
            "ieeg_json_member": record.ieeg_json_member,
            "coordsystem_member": record.coordsystem_member,
            "electrodes_member": record.electrodes_member,
            "preprocessing_fully_causal": self.resampling_mode == "causal_upfirdn",
            "bandpass_causal": True,
            "filter_type": "Butterworth SOS (forward only)",
            "filter_order": self.filter_order,
            "bandpass_hz": list(self.bandpass_hz),
            "filter_burn_in_s_requested": self.burn_in_s,
            "filter_burn_in_s_applied": burn_samples / source_sfreq,
            "reference": "common average over selected status=good channels",
            "resampling": (
                f"scipy.signal.upfirdn causal FIR up={ratio.numerator} "
                f"down={ratio.denominator}"
                if self.resampling_mode == "causal_upfirdn"
                else f"scipy.signal.resample_poly up={ratio.numerator} down={ratio.denominator}"
            ),
            "resampling_mode": self.resampling_mode,
            "resampling_causal": self.resampling_mode == "causal_upfirdn",
            "resampling_kernel": (
                "causal Kaiser-window anti-aliasing FIR with retained group delay"
                if self.resampling_mode == "causal_upfirdn"
                else "symmetric anti-aliasing FIR"
            ),
            "causal_fir_half_length_factor": self.causal_fir_half_length_factor,
            "causal_fir_group_delay_s": causal_group_delay_s,
            "forecast_boundary_guard_required": self.resampling_mode != "causal_upfirdn",
            "forecast_boundary_guard_s": guard_s,
            "forecast_boundary_guard_samples": guard_samples,
            "event_onset_s": metadata.onset_s,
            "raw_event_offset_s": metadata.raw_offset_s,
            "effective_event_offset_s": metadata.effective_offset_s,
            "offset_qc_override": metadata.offset_qc is not None,
        }
        return ProcessedSegment(
            data=np.asarray(resampled, dtype=np.float32),
            channel_names=channel_names,
            status_descriptions=descriptions,
            coordinates=coordinates,
            coordinate_units=coordinate_units,
            subject=record.subject,
            task=record.task,
            run=record.run,
            start_s=start_s,
            duration_s=duration_s,
            source_sfreq=source_sfreq,
            sfreq=self.target_sfreq,
            qc_notes=tuple(notes),
            provenance=provenance,
        )


def load_patient_segments(
    loader: EEGSegmentLoader,
    subject: str,
    ictal_runs: Sequence[str | int],
    interictal_runs: Sequence[str | int],
    ictal_duration_s: float = 24.0,
    interictal_start_s: float = 60.0,
    interictal_duration_s: float = 30.0,
    use_cache: bool = True,
) -> dict[str, ProcessedSegment]:
    """Load selected runs on one deterministic common-good-channel basis."""

    selections = [*(('ictal', run) for run in ictal_runs), *(('interictal', run) for run in interictal_runs)]
    records = loader.dataset.selected_records(subject, selections)
    common = loader.dataset.common_good_channels(subject, records=records)
    output: dict[str, ProcessedSegment] = {}
    for run in ictal_runs:
        segment = loader.load_ictal(
            subject,
            run,
            duration_s=ictal_duration_s,
            channel_names=common,
            use_cache=use_cache,
        )
        output[f"ictal_{segment.run}"] = segment
    for run in interictal_runs:
        segment = loader.load_interictal(
            subject,
            run,
            start_s=interictal_start_s,
            duration_s=interictal_duration_s,
            channel_names=common,
            use_cache=use_cache,
        )
        output[f"interictal_{segment.run}"] = segment
    return output


__all__ = [
    "BIDSRecord",
    "BIDSZipDataset",
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_SUBJECTS",
    "EEGSegmentLoader",
    "KNOWN_OFFSET_QC",
    "OffsetQC",
    "ProcessedSegment",
    "RunMetadata",
    "load_patient_segments",
]
