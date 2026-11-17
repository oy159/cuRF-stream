import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from room_evaluation import evaluate_case, strict_match, summarize
from signal_policy import FAMILIES, SignalPolicy, load_config


class RoomEvaluationTests(unittest.TestCase):
    def test_matching_isolates_receivers_and_rejects_off_centre_skirts(self):
        e={"stream":1,"fc":1e6,"bw":1e6,"t0":0,"t1":1}
        r={"s":1,"f":1e6,"bw":1e6,"t0":.1,"t1":.2}
        self.assertGreater(strict_match(r,e),0)
        self.assertEqual(strict_match({**r,"s":0},e),0)
        self.assertEqual(strict_match({**r,"f":1.4e6},e),0)
        packet={**e,"t0":.30712,"t1":.30747}
        self.assertGreater(strict_match({**r,"t0":.30663,"t1":.30716},packet),0)
        self.assertEqual(strict_match({**r,"t0":.305,"t1":.306},packet),0)
        # A tracked transmitter's long-term occupied envelope may cover hopped
        # packets; the packet itself must still intersect this observation.
        hopped={**e,"fc":1.8e6,"carrier_fc":1e6,"carrier_bw":4e6}
        self.assertGreater(strict_match({**r,"bw":3e6},hopped),0)

    def score(self,stream=0,protocol="wifi",family="ofdm",complete=True,processed=False,mismatch=False):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            truth={"version":2,"seed":1,"profile":"ism24","kind":"positive","nominal_snr_db":[20,30],"freq":2441e6,
                   "streams":2,"secs":.8,"generation":{"clipped_fraction":0},
                   "carriers":[{"source_id":"tx","expected_named":True,"snr_db":20}],
                   "truth":[{"stream":stream,"fc":1e6,"bw":18e6,"t0":0,"t1":.8,"family":family,
                             "protocol":protocol,"expected_named":True,"source_id":"tx"}]}
            probs=[0.]*13
            probs[0],probs[1]=.01,.99
            rows=[{"meta":{"rate":61.44e6,"freq":2441e6,"streams":2,"signal_classes":FAMILIES}},
                  {"sig":{"w":1,"s":0,"f":1e6,"bw":18e6,"t":.1,"t0":.09,"t1":.1,"c":1,"probs":probs,"activity_fraction":.2,"snr_db":25}}]
            if complete:
                rows.append({"end":{}})
            config=load_config()
            config["confirmations"]=1
            if processed:
                p=SignalPolicy(config)
                rows=[p.process(row) for row in rows]
                if mismatch:
                    rows[0]["meta"]["policy"]["config"]={**config,"confirmations":2}
            (root/"truth.json").write_text(json.dumps(truth))
            (root/"raw.feed").write_text("\n".join(json.dumps(r) for r in rows))
            (root/"runtime.json").write_text(json.dumps({"wall_s":.8,"dropped_frames":0,"gap_frames":0,
                "stage2":{"skipped_busy":0,"skipped_ring":0,"skipped_capacity":0}}))
            return evaluate_case(root,config,processed=processed)

    def test_correct_family_with_wrong_protocol_is_a_label_error(self):
        r=self.score(protocol="lte")
        self.assertEqual(r["counts"]["protocol_wrong"],1)
        self.assertEqual(r["counts"]["errors"],1)
        self.assertEqual(r["errors"][0]["reason"],"wrong_protocol")

    def test_wrong_family_and_protocol_count_as_one_error(self):
        r=self.score(protocol="ble",family="fsk")
        self.assertEqual(r["counts"]["errors"],1)
        self.assertEqual(r["counts"]["protocol_wrong"],1)

    def test_other_rx_cannot_supply_a_correct_label_match(self):
        r=self.score(stream=1)
        self.assertEqual(r["counts"]["unmatched"],1)

    def test_true_label_is_named_and_incomplete_capture_cannot_pass(self):
        r=self.score()
        self.assertEqual(r["counts"]["correct"],1)
        self.assertEqual(r["high_snr_named"],1)
        with self.assertRaisesRegex(ValueError,"incomplete"):
            self.score(complete=False)

    def test_processed_decisions_are_scored_with_the_same_frozen_policy(self):
        self.assertEqual(self.score(), self.score(processed=True))
        with self.assertRaisesRegex(ValueError,"policy does not match"):
            self.score(processed=True,mismatch=True)

    def test_rejecting_everything_cannot_pass_acceptance(self):
        runs=[{"counts":{"unknown":100},"profile":p,"checks":{"complete":True},"kind":"positive", "high_snr_named":0,"high_snr_carriers":1}
              for p in ("ism24","wlan5","cell900","cell2100")]
        r=summarize(runs,"acceptance")
        self.assertFalse(r["passed"])
        self.assertIsNone(r["totals"]["error_fraction"])
        self.assertFalse(r["checks"]["enough_named_per_band"])


if __name__=="__main__":
    unittest.main()
