import sys
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from check_dsss_capture import BARKER, coherent_energy, inspect


class DsssCaptureTests(unittest.TestCase):
    def test_barker_survives_data_polarity_timing_and_carrier_offset(self):
        rng = np.random.default_rng(14)
        bits = rng.choice([-1, 1], 1024)
        x = np.repeat((bits[:, None]*BARKER[::-1]).ravel(), 2).astype(np.complex64)
        x *= np.exp(2j*np.pi*30e3/22e6*np.arange(len(x)))
        x += .1*(rng.normal(size=len(x))+1j*rng.normal(size=len(x)))
        x = np.r_[np.zeros(7, np.complex64), x]
        r = coherent_energy(x, 22e6)
        self.assertGreater(r["max_coherent_fraction"], .9)
        self.assertEqual(r["above_0p5"], r["windows_64us"])

    def test_noise_is_not_matched_by_timing_search(self):
        rng = np.random.default_rng(15)
        x = (rng.normal(size=22528)+1j*rng.normal(size=22528)).astype(np.complex64)
        self.assertEqual(coherent_energy(x, 22e6)["above_0p5"], 0)

    def test_silence_is_finite(self):
        r = coherent_energy(np.zeros(22528, np.complex64), 22e6)
        self.assertEqual(r["max_coherent_fraction"], 0)
        self.assertEqual(r["above_0p5"], 0)

    def test_short_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "microseconds"):
            coherent_energy(np.zeros(64, np.complex64), 22e6)

    def test_generator_sigmf_without_rf_frequency(self):
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory)/"wifi"
            np.tile(np.repeat(BARKER, 2), 1024).astype("<c8").tofile(stem.with_suffix(".cf32"))
            meta = stem.with_suffix(".sigmf-meta")
            meta.write_text(json.dumps({"global":{"core:sample_rate":22e6,"core:datatype":"cf32_le"},
                                        "captures":[{"core:sample_start":0}]}))
            r = inspect(meta)
            self.assertEqual(r["frequency"], 0)
            self.assertEqual(len(r["data_sha256"]), 64)
            self.assertGreater(r["max_coherent_fraction"], .99)
