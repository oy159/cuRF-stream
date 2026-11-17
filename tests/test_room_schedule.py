from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_room_checks import cases


class RoomScheduleTests(unittest.TestCase):
    def test_acceptance_is_separate_and_balances_bands_snr_and_receivers(self):
        dev, acceptance = cases("dev", 32), cases("acceptance", 96)
        self.assertFalse({c["seed"] for c in dev} & {c["seed"] for c in acceptance})
        self.assertEqual(len(acceptance), 104)
        self.assertEqual(len({c["seed"] for c in acceptance}), 104)
        positives = [c for c in acceptance if c["kind"] == "positive"]
        counts = Counter((c["profile"], tuple(c["snr"])) for c in positives)
        self.assertEqual(len(counts), 16)
        self.assertEqual(set(counts.values()), {6})
        self.assertEqual({c["streams"] for c in positives}, {1, 2})
        dual = {(c["profile"], tuple(c["snr"])) for c in positives if c["streams"] == 2}
        self.assertEqual(dual, set(counts))
        controls = Counter((c["kind"], c["profile"]) for c in acceptance if c["kind"] != "positive")
        self.assertEqual(len(controls), 8)
        self.assertEqual(set(controls.values()), {1})

    def test_revalidation_uses_a_new_explicit_seed_range(self):
        first, second = cases("acceptance",96), cases("acceptance",96,33001)
        self.assertFalse({c["seed"] for c in first} & {c["seed"] for c in second})
        self.assertEqual(second[0]["seed"],33001)
        self.assertEqual(second[-1]["seed"],34008)


if __name__ == "__main__":
    unittest.main()
