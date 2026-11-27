"""Signal-free tests of cache role, preprocessing and channel-order validation."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_config import PATIENTS, PREPROCESSING, Replay, check_segment


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.names = [f"test-{i:02}" for i in range(64)]
        self.meta = dict(subject="HUP065", run="run-01", start_s=120., duration_s=32.,
                         sfreq=256., source_sfreq=512., units="uV", task="ictal",
                         provenance=dict(PREPROCESSING))
        digest = sha256("\n".join(self.names).encode()).hexdigest()
        patched = patch.dict(PATIENTS, {"HUP065": (64, 32, digest)})
        patched.start()
        self.addCleanup(patched.stop)

    def case(self, meta=None, names=None, shape=(64, 8192)):
        buffer = BytesIO()
        np.savez_compressed(buffer, data=np.zeros(shape, dtype="<f4"),
                            channel_names=self.names if names is None else names,
                            metadata_json=json.dumps(self.meta if meta is None else meta))
        buffer.seek(0)
        return Replay("HUP065", Path("."), Path("."), buffer, None, None)

    def test_valid_metadata(self):
        self.assertEqual(check_segment(self.case(), "run01_ictal")["shape"], [64, 8192])

    def test_disallow_other_run(self):
        with self.assertRaisesRegex(ValueError, "expected run"):
            check_segment(self.case(dict(self.meta, run="run-02")), "run01_ictal")

    def test_disallow_noncausal_resampling(self):
        self.meta["provenance"]["resampling_mode"] = "symmetric_polyphase"
        with self.assertRaisesRegex(ValueError, "resampling_mode"):
            check_segment(self.case(), "run01_ictal")

    def test_disallow_reordered_or_truncated_inputs(self):
        with self.assertRaisesRegex(ValueError, "channel basis"):
            check_segment(self.case(names=self.names[::-1]), "run01_ictal")
        with self.assertRaisesRegex(ValueError, "shape"):
            check_segment(self.case(shape=(64, 8191)), "run01_ictal")
