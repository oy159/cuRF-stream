import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from signal_policy import FAMILIES, SignalPolicy, calibration_context, load_config


def probabilities(family, confidence=.97):
    p = [0.] * len(FAMILIES)
    p[FAMILIES.index(family)] = confidence
    p[0 if family != "noise" else -1] = 1-confidence
    return p


def record(w=0, family="ofdm", **kwargs):
    p = probabilities(family)
    return {"s": 0, "id": 0, "w": w, "f": 1e6, "bw": 18e6, "t": w*.03,
            "t0": w*.03-.001, "t1": w*.03, "c": FAMILIES.index(family), "p": max(p), "probs": p,
            "activity_fraction": .25, "snr_db": 25, **kwargs}


def policy(freq=2441e6, **kwargs):
    p = SignalPolicy(**kwargs)
    p.process({"meta": {"freq": freq, "rate": 61.44e6, "streams": 2, "channels": 512, "window": 1024,
                         "signal_classes": FAMILIES, "source": "file"}})
    return p


class SignalPolicyTests(unittest.TestCase):
    def test_prior_retains_raw_mass_and_does_not_manufacture_confidence(self):
        p = policy(5500e6)
        probs = probabilities("fm", .97)
        probs[FAMILIES.index("ofdm")] = .02
        probs[0] = .01
        r = record(probs=probs)
        before = copy.deepcopy(r)
        for w in range(10):
            result = p.process({"sig": {**r, "w": w, "t": .03*w}})
        d = result["display"]
        self.assertEqual(d["family"], "unknown")
        self.assertEqual(d["posterior"], 1)
        self.assertAlmostEqual(d["allowed_mass"], .02)
        self.assertEqual(d["reason"], "allowed_mass")
        self.assertEqual(r, before)

    def test_disallowed_families_never_show_in_ism(self):
        for family in ("fm", "am", "psk", "qam", "unknown", "noise"):
            p = policy()
            for w in range(5):
                d = p.process({"sig": record(w, family)})["display"]
                self.assertEqual(d["family"], "unknown")

    def test_confirmation_and_release_hysteresis(self):
        p = policy()
        ds = [p.process({"sig": record(w)})["display"] for w in range(3)]
        self.assertEqual([d["status"] for d in ds], ["unknown", "unknown", "stable"])
        self.assertEqual(ds[-1]["protocol"], "WiFi（推测）")
        d = p.process({"sig": record(3, probs=probabilities("ofdm", .87))})["display"]
        self.assertEqual(d["status"], "held")
        d = p.process({"sig": record(4, "fm")})["display"]
        self.assertEqual(d["family"], "unknown")
        # Conflicting evidence is exposed immediately, and expires the stable state.
        p.process({"sig": record(5, "fm")})
        self.assertIsNone(next(iter(p.carriers.values())).stable)

    def test_vote_by_frequency_survives_short_burst_track_ids(self):
        p = policy()
        for i, t in enumerate((0, 1, 2)):
            d = p.process({"sig": record(i, "fsk", id=i+42, bw=1e6, t=t)})["display"]
        self.assertEqual(d["family"], "fsk")
        self.assertEqual(d["uid"], 1)
        self.assertEqual(d["protocol"], "蓝牙 / BLE（推测）")
        self.assertEqual(p.process({"sig": record(4, "fsk", t=6, bw=1e6)})["display"]["status"], "unknown")

    def test_raw_id_reuse_and_streams_do_not_share_votes(self):
        p = policy()
        for w in range(3):
            p.process({"sig": record(w)})
        self.assertEqual(p.process({"sig": record(4, s=1)})["display"]["status"], "unknown")
        self.assertEqual(p.process({"sig": record(5, f=-15e6)})["display"]["status"], "unknown")

    def test_duplicate_hop_does_not_count_as_three_confirmations(self):
        p = policy()
        for _ in range(10):
            d = p.process({"sig": record()})["display"]
        self.assertEqual((d["status"], d["confirmations"]), ("unknown", 1))

    def test_baseband_uses_stricter_threshold_and_no_protocol_guess(self):
        p = policy(0)
        for w in range(5):
            d = p.process({"sig": record(w, probs=probabilities("ofdm", .9))})["display"]
        self.assertEqual(d["family"], "unknown")
        for w in range(5, 8):
            d = p.process({"sig": record(w)})["display"]
        self.assertEqual(d["family"], "ofdm")
        self.assertIsNone(d["protocol"])

    def test_probabilities_must_be_complete_finite_and_normalized(self):
        p = policy()
        for probs in (None, [.99], [float("nan")]*13, [.9]*13, [-.1]+[1.1]+[0.]*11):
            d = p.process({"sig": record(probs=probs)})["display"]
            self.assertEqual(d["family"], "unknown")
            self.assertEqual(d["reason"], "missing_probabilities")

    def test_small_margin_abstains(self):
        cfg = load_config()
        cfg.update(confidence=.5, raw_confidence=.45, margin=.2)
        p = policy(config=cfg)
        probs = [0.]*13
        probs[1], probs[2] = .52, .48
        for w in range(5):
            d = p.process({"sig": record(w, probs=probs)})["display"]
        self.assertEqual(d["reason"], "small_margin")

    def test_protocol_requires_timing_and_does_not_claim_nr_from_shared_band(self):
        p = policy(2600e6)
        for i in range(4):
            d = p.process({"sig": record(i, t=i*.2, bw=5e6, activity_fraction=1)})["display"]
        self.assertEqual(d["protocol"], "LTE / NR（推测）")
        p = policy()
        for i in range(4):
            d = p.process({"sig": record(i, activity_fraction=-1)})["display"]
        self.assertIsNone(d["protocol"])

    def test_1090_is_decoder_only_and_crc_is_required(self):
        p = policy(1080e6)
        d = p.process({"sig": record(f=10e6, bw=2e6)})["display"]
        self.assertEqual((d["status"], d["visible"]), ("decoder_only", False))
        a = {"s": 1, "freq": 1090e6, "df": 17, "icao": "ABC123", "t": .1}
        self.assertTrue(p.process({"adsb": a})["adsb_rejected"])
        result = p.process({"adsb": {**a, "crc_ok": True}})
        self.assertEqual(result["display"]["s"], 1)
        self.assertEqual(result["display"]["f"], 10e6)
        self.assertIsNone(result["display"]["confidence"])
        self.assertTrue(p.process({"adsb": {**a, "crc_ok": True, "df": 11}})["adsb_rejected"])

    def test_gsm_guess_is_limited_to_900_and_1800_downlink(self):
        for freq, expected in ((939e6, "GSM（推测）"), (1835e6, "GSM（推测）"), (2135e6, None), (2350e6, None)):
            p = policy(freq)
            for w in range(4):
                d = p.process({"sig": record(w, "fsk", t=w*.2, bw=200e3, activity_fraction=1)})["display"]
            self.assertEqual(d["family"], "fsk")
            self.assertEqual(d["protocol"], expected)

    def test_spur_requires_exact_receiver_context_and_does_not_mask_wide_neighbor(self):
        p = policy()
        meta = {**p.meta, "source": "omnicore", "rx": [1, 2], "receiver": {"serial": "test", "gain": 30}}
        table = {"context": calibration_context(meta), "entries": [{"stream": 0, "lo_hz": .9e6, "hi_hz": 1.1e6}]}
        p.spurs = table
        p.process({"meta": meta})
        self.assertEqual(p.process({"sig": record(bw=100e3)})["display"]["status"], "spur")
        self.assertEqual(p.process({"sig": record(1, bw=18e6)})["display"]["status"], "unknown")
        p.process({"meta": {**meta, "freq": 5500e6}})
        self.assertEqual(p.spur_status, "context_mismatch")
        self.assertEqual(p.process({"sig": record(bw=100e3)})["display"]["status"], "unknown")

    def test_reset_on_identical_meta_and_bounded_state(self):
        p = policy()
        for w in range(3):
            p.process({"sig": record(w)})
        p.process({"meta": p.meta})
        self.assertEqual(p.stats()["records"], 0)
        self.assertEqual(p.process({"sig": record()})["display"]["status"], "unknown")
        p.config["max_carriers"] = 4
        for i in range(30):
            p.process({"sig": record(i, f=i*1e6, bw=1e3)})
        self.assertEqual(len(p.carriers), 4)

    def test_unknown_fraction_is_separate_and_undefined_without_records(self):
        p = policy()
        self.assertIsNone(p.stats()["unknown_fraction"])
        for w in range(3):
            p.process({"sig": record(w)})
        self.assertAlmostEqual(p.stats()["unknown_fraction"], 2/3)
        self.assertEqual(p.process({"end": {}})["end"]["policy"]["named"], 1)

    def test_signal_quality_gates_cannot_be_bypassed_by_confidence_or_hysteresis(self):
        p=policy()
        for w in range(3):
            p.process({"sig":record(w)})
        d=p.process({"sig":record(3,snr_db=3)})["display"]
        self.assertEqual((d["family"],d["reason"]),("unknown","low_snr"))
        d=p.process({"sig":record(4,snr_db=None)})["display"]
        self.assertEqual(d["reason"],"missing_snr")
        for w in range(5,10):
            d=p.process({"sig":record(w,"fsk",bw=12e3,snr_db=30)})["display"]
        self.assertEqual((d["family"],d["reason"]),("unknown","bandwidth_prior"))
        for w in range(10,13):
            d=p.process({"sig":record(w,"fsk",bw=1e6,snr_db=25)})["display"]
        self.assertEqual(d["family"],"fsk")

    def test_wide_wifi_family_off_channel_grid_does_not_keep_a_stable_label(self):
        p=policy()
        for w in range(3):
            d=p.process({"sig":record(w)})["display"]
        self.assertEqual(d["family"],"ofdm")
        d=p.process({"sig":record(3,f=1.8e6)})["display"]
        self.assertEqual((d["family"],d["reason"]),("unknown","off_channel_grid"))

    def test_nearby_wide_component_prevents_naming_a_narrow_piece_in_either_order(self):
        for narrow_first in (False,True):
            p=policy()
            for w in range(3):
                narrow={"sig":record(w,"dsss",f=11.5e6,bw=2e6)}
                wide={"sig":record(w)}
                if narrow_first:
                    d=p.process(narrow)["display"]
                    p.process(wide)
                else:
                    p.process(wide)
                    d=p.process(narrow)["display"]
            self.assertEqual((d["family"],d["reason"]),("unknown","near_wideband_component"))

    def test_wide_component_guard_is_local_to_receiver_and_expires(self):
        for different_rx in (False,True):
            p=policy()
            p.process({"sig":record(0)})
            start=1 if different_rx else 10
            for w in range(start,start+3):
                d=p.process({"sig":record(w,"dsss",s=int(different_rx),f=11.5e6,bw=2e6)})["display"]
            self.assertEqual(d["family"],"dsss")


if __name__ == "__main__":
    unittest.main()
