import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from make_scene import match, score, refine_times
import numpy as np


class SceneScoreTests(unittest.TestCase):
    def setUp(self):
        self.truth = {"scene": 0, "family": "ofdm", "fc": 1e6, "bw": 1e6, "t0": .1, "t1": .2}
        self.record = {"f": 1e6, "bw": 1e6, "t0": .12, "t1": .13, "c": 1, "p": .95}

    def test_matching_requires_frequency_and_time(self):
        self.assertGreater(match(self.record, self.truth), 0)
        self.assertEqual(match({**self.record, "t0": 0, "t1": .01}, self.truth), 0)
        self.assertEqual(match({**self.record, "f": 5e6}, self.truth), 0)
        self.assertEqual(match({**self.record, "bw": 20e6}, self.truth), 0)

    def test_short_burst_inside_long_window_can_match(self):
        self.assertGreater(match({**self.record, "t0": .09, "t1": .21}, self.truth), 0)

    def test_truth_uses_clean_sample_edges_instead_of_stft_padding(self):
        x = np.zeros(100, np.complex64)
        x[32:41] = 1
        e = {"t0": 0, "t1": 100, "fc": 0, "bw": 1}
        r = refine_times(x, e)
        self.assertEqual((r["t0"], r["t1"]), (32, 41))
        self.assertIsNone(refine_times(x * 0, e))

    def test_scoring_counts_wrong_class_and_spurious_records(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "truth.json").write_text(json.dumps({"version": 2, "scenes": 1, "truth": [self.truth]}))
            records = [{"sig": self.record}, {"sig": {**self.record, "c": 2}},
                       {"sig": {**self.record, "t0": .3, "t1": .31}}, {"end": {}}]
            (root / "scene_0.feed").write_text("\n".join(json.dumps(r) for r in records))
            with contextlib.redirect_stdout(io.StringIO()):
                score(SimpleNamespace(out=root, confidence=0))
            result = json.loads((root / "score.json").read_text())
            self.assertEqual((result["correct"], result["wrong_class"], result["unmatched"]), (1, 1, 1))
            self.assertEqual(result["reported_emissions"], 1)
            self.assertAlmostEqual(result["error_fraction"], 2/3)
            (root / "scene_0.feed").write_text(json.dumps({"sig": self.record}))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                score(SimpleNamespace(out=root, confidence=0))


if __name__ == "__main__":
    unittest.main()
