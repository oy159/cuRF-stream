import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from check_frequency_evidence import evaluate
from signal_policy import FAMILIES, load_config


class FrequencyEvidenceTests(unittest.TestCase):
    def feed(self):
        rows = [{"meta": {"streams": 2, "rate": 61.44e6, "freq": 2450e6, "signal_classes": FAMILIES}}]
        for stream in (0, 1):
            for i in range(3):
                rows.append({"sig": {"w": i*6, "s": stream, "id": 3, "f": 12e6, "extraction_f": 12.9e6,
                    "bw": 19e6, "fs": 46.08e6, "t": .1+i*.03, "t0": .1+i*.03, "t1": .10003+i*.03,
                    "c": 6, "p": .99, "probs": [.01 if k==0 else .99 if k==6 else 0 for k in range(13)],
                    "db": -12, "snr_db": 20, "activity_fraction": .4,
                    "dsss_frequency": {"offset_hz": -900000, "coherence": .86, "contrast": .2, "retained_power": .98, "valid": True}}})
        return rows+[{"end": {}}]

    def score(self, root, rows):
        p = root/"raw.feed"
        p.write_text("".join(json.dumps(r)+"\n" for r in rows))
        return p, evaluate(p, load_config(), 1., 1, [2462e6])

    def test_same_observations_preserve_rx_and_raw_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.feed()
            original = copy.deepcopy(rows)
            p, r = self.score(Path(tmp), rows)
            self.assertEqual(rows, original)
            self.assertEqual(r["feed_sha256"], hashlib.sha256(p.read_bytes()).hexdigest())
            on, off = (r["results"][k] for k in ("corrected", "extraction_centres"))
            self.assertEqual(on["policy"]["records"], off["policy"]["records"])
            self.assertEqual(on["policy"]["named"], 2)
            self.assertEqual(off["policy"]["named"], 0)
            self.assertEqual(on["targets"]["2462.0"]["named"], 1)
            self.assertEqual(off["targets"]["2462.0"]["named"], 0)
            self.assertEqual(off["targets"]["2462.0"]["maximum_gap_s"], 1.)

    def test_incomplete_feed_and_unsupported_correction_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                self.score(root, self.feed()[:-1])
            rows = self.feed()
            rows[1]["sig"]["dsss_frequency"]["coherence"] = .3
            with self.assertRaisesRegex(ValueError, "structural evidence"):
                self.score(root, rows)


if __name__ == "__main__":
    unittest.main()
