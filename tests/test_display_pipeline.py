import json
from pathlib import Path
import queue
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from apply_signal_policy import apply, DisplayScore
from build_spur_table import build
from signal_policy import FAMILIES, SignalPolicy, load_spurs
from ui_server import Hub, run_live


def signal(w, **kwargs):
    p = [0.]*13
    p[0], p[1] = .02, .98
    return {"sig": {"s": 0, "id": 2, "w": w, "f": 1e6, "bw": 18e6, "t": w*.03,
                    "t0": w*.03, "t1": w*.03+.001, "c": 1, "p": .98, "probs": p, **kwargs}}


META = {"meta": {"source": "file", "stage2": True, "freq": 2440e6, "rate": 61.44e6,
                 "streams": 1, "channels": 512, "window": 1024, "every": 8, "signal_classes": FAMILIES}}


class DisplayPipelineTests(unittest.TestCase):
    def test_live_and_offline_produce_identical_decisions(self):
        raw = [META, *[signal(i) for i in range(5)], {"end": {}}]
        hub = Hub()
        live = [json.loads(hub.publish(json.dumps(r))) for r in raw]
        with tempfile.TemporaryDirectory() as d:
            feed, out = Path(d)/"in.feed", Path(d)/"out.feed"
            feed.write_text("\n".join(json.dumps(r) for r in raw))
            apply(feed, out, SignalPolicy())
            offline = [json.loads(l) for l in out.read_text().splitlines()]
            self.assertEqual(live, offline)
            for before, after in zip(raw[1:-1], offline[1:-1]):
                self.assertEqual(before["sig"], after["sig"])
            replay = Hub()
            self.assertEqual([json.loads(replay.publish(json.dumps(r))) for r in offline], offline)

    def test_live_child_pipe_and_tee_preserve_raw_and_policy(self):
        with tempfile.TemporaryDirectory() as d:
            tee = Path(d)/"live.feed"
            rows = [META, signal(0), signal(1), signal(2), {"end": {}}]
            code = ("import sys; f=open(sys.argv[-1], 'w'); "
                    + "f.write(" + repr("\n".join(json.dumps(r) for r in rows)+"\n") + "); f.close()")
            run_live(Hub(), [sys.executable, "-c", code], tee)
            output = [json.loads(line) for line in tee.read_text().splitlines()]
            self.assertEqual(output[3]["display"]["status"], "stable")
            self.assertEqual(output[3]["sig"], rows[3]["sig"])
            self.assertEqual(output[-1]["end"]["policy"]["named"], 1)

    def test_finite_scoring_separates_unknown_and_false_named(self):
        truth = {"version": 2, "truth": [{"scene": 0, "family": "ofdm", "fc": 1e6, "bw": 18e6, "t0": 0, "t1": .5}]}
        score = DisplayScore(truth)
        self.assertIsNone(score.result()["named_error_fraction"])
        p = SignalPolicy()
        p.process(META)
        for w in range(5):
            score.add(p.process(signal(w)))
        result = score.result()
        self.assertEqual((result["named"], result["unknown"], result["correct"]), (3, 2, 3))
        self.assertEqual(result["named_error_fraction"], 0)
        self.assertEqual(result["unknown_fraction"], .4)
        # A named result at an absent time is an error, even if the family agrees.
        row = p.process(signal(6, t0=1, t1=1.01))
        score.add(row)
        self.assertEqual(score.result()["unmatched"], 1)

    def test_incomplete_feeds_and_in_place_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            feed, out = Path(d)/"in.feed", Path(d)/"out.feed"
            feed.write_text(json.dumps(META))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                apply(feed, out, SignalPolicy())
            with self.assertRaisesRegex(ValueError, "different"):
                apply(feed, feed, SignalPolicy())

    def test_late_browser_receives_bounded_latest_state_and_loop_resets(self):
        hub = Hub()
        hub.publish(json.dumps(META))
        for w in range(1100):
            hub.publish(json.dumps(signal(w)))
        hub.publish('{"end":{}}')
        q = hub.subscribe()
        rows = []
        while not q.empty():
            rows.append(json.loads(q.get_nowait()))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[1]["display"]["status"], "stable")
        self.assertIn("end", rows[-1])
        # Identical metadata is a new run; no stale history or accumulated votes.
        hub.publish(json.dumps(META))
        self.assertEqual(hub.policy.stats()["records"], 0)
        self.assertEqual(q.qsize(), 1)
        hub.unsubscribe(q)

    def test_calibration_requires_completed_manual_gain_hardware_capture(self):
        with tempfile.TemporaryDirectory() as d:
            feed = Path(d)/"cal.feed"
            meta = {"meta": {"source": "omnicore", "freq": 98e6, "rate": 8e6, "streams": 1, "channels": 8,
                             "rx": [1], "receiver": {"device": "test board", "agc": 0, "gain": 30}}}
            power = [-80]*8
            power[0] = -30  # DC, in unshifted channel order
            rows = [meta] + [{"t": i*.2, "p": power, "f": [-80]} for i in range(65)]
            rows += [{"end": {"dropped_frames": 0, "gap_frames": 0}}]
            feed.write_text("\n".join(json.dumps(r) for r in rows))
            table = build(feed)
            self.assertEqual(table["entries"], [{"stream": 0, "lo_hz": -.5e6, "hi_hz": .5e6, "label": "本机杂散"}])
            table_path = Path(d)/"spurs.json"
            table_path.write_text(json.dumps(table))
            self.assertEqual(load_spurs(table_path), table)
            rows[0]["meta"]["receiver"]["agc"] = 1
            feed.write_text("\n".join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, "manual gain"):
                build(feed)
            table["termination"] = "antenna"
            table_path.write_text(json.dumps(table))
            with self.assertRaisesRegex(ValueError, "50ohm"):
                load_spurs(table_path)

    def test_snapshot_does_not_duplicate_decoder_event_or_lose_power_history(self):
        hub = Hub()
        hub.publish(json.dumps(META))
        for i in range(300):
            hub.publish(json.dumps({"t": i*.01, "w": i, "p": [-80]*512, "f": [-80], "r": []}))
        hub.publish(json.dumps({"adsb": {"s": 0, "freq": 1090e6, "df": 17, "icao": "40621D", "t": 3, "crc_ok": True}}))
        q = hub.subscribe()
        rows = []
        while not q.empty():
            rows.append(json.loads(q.get_nowait()))
        self.assertEqual(sum("adsb" in r for r in rows), 1)
        self.assertEqual(sum("p" in r for r in rows), 250)
        hub.unsubscribe(q)


if __name__ == "__main__":
    unittest.main()
