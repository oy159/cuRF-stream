import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from live_feed import DisplayReducer, capture_file_feed
from signal_policy import FAMILIES, SignalPolicy
from ui_server import Hub


META = {"meta":{"freq":2441e6,"rate":61.44e6,"streams":2,"signal_classes":FAMILIES}}


def signal(t, stream=0, track=1, p=.95):
    probs = [0.]*len(FAMILIES)
    probs[0], probs[1] = 1-p, p
    return {"sig":{"s":stream,"id":track,"t":t,"w":round(t*1000),"t0":t-.001,"t1":t,
                   "f":1e6+track*20e6,"bw":18e6,"snr_db":25,"activity_fraction":.2,"probs":probs}}


class DisplayReducerTests(unittest.TestCase):
    def test_latest_observation_keeps_original_probabilities_and_source_order(self):
        r = DisplayReducer(.2)
        r.push(META)
        self.assertEqual(r.push(signal(0,p=.5)), [])
        self.assertEqual(r.push(signal(.1,p=.6)), [])
        rows = r.push(signal(.21,p=.7)) + r.push({"end":{}})
        self.assertEqual([x["sig"]["t"] for x in rows if "sig" in x], [.1,.21])
        self.assertEqual(rows[0]["sig"], signal(.1,p=.6)["sig"])
        self.assertEqual(r.stats()["coalesced_signals"], 1)

    def test_receivers_are_separate_and_every_decoder_message_survives(self):
        r = DisplayReducer(.2)
        r.push(META)
        r.push(signal(0,0)); r.push(signal(.01,1))
        a,b = {"adsb":{"crc_ok":True,"message":1}}, {"adsb":{"crc_ok":True,"message":2}}
        rows = r.push(a)+r.push(b)
        self.assertEqual([x["sig"]["s"] for x in rows if "sig" in x], [0,1])
        self.assertEqual([x["adsb"] for x in rows if "adsb" in x], [a["adsb"],b["adsb"]])

    def test_capacity_flushes_without_discarding_distinct_tracks(self):
        r = DisplayReducer(1,max_pending=2)
        r.push(META)
        rows = []
        for i in range(5): rows += r.push(signal(i*.01,track=i))
        rows += r.flush()
        self.assertEqual(len(rows), 5)
        self.assertEqual(r.stats()["coalesced_signals"], 0)
        self.assertLessEqual(r.stats()["peak_pending"], 2)

    def test_every_power_row_survives_and_flushes_delayed_signal_updates(self):
        r = DisplayReducer(.2)
        r.push(META)
        r.push(signal(0))
        powers = [{"t":i*.04,"p":[i]} for i in range(10)]
        rows = []
        for p in powers: rows += r.push(p)
        self.assertEqual([x for x in rows if "p" in x], powers)
        self.assertEqual([x["sig"] for x in rows if "sig" in x], [signal(0)["sig"]])

    def test_short_confirmed_burst_survives_later_unknown_and_counters_cover_all_input(self):
        r = DisplayReducer(.2)
        rows = r.push(META)
        observations = [signal(t,track=0) for t in (0,.03,.06,.09)] + [signal(.12,track=0,p=.1)]
        reference = SignalPolicy()
        reference.process(META)
        decisions = [reference.process(x) for x in observations]
        for x in observations: rows += r.push(x)
        rows += r.push({"end":{}})
        signals = [x for x in rows if "sig" in x]
        self.assertEqual([x["display"]["family"] for x in signals], ["ofdm","ofdm","unknown"])
        self.assertEqual(signals, decisions[2:])
        self.assertEqual(rows[-1]["end"]["policy"], reference.stats())
        self.assertEqual(r.stats()["input_signals"], 5)


@unittest.skipUnless(shutil.which("prlimit"), "Linux file quota integration")
class FileFeedTests(unittest.TestCase):
    def run_fake(self, root, budget):
        fake = root/"receiver"
        input_rows = [META]+[signal(i/1000,p=.5) for i in range(300)]+[{"end":{}}]
        fake.write_text("#!"+sys.executable+"\nimport json,sys\nf=open(sys.argv[-1],'w')\n"
                        +"for row in "+repr(input_rows)+": f.write(json.dumps(row)+'\\n')\n"
                        +"f.close()\nprint('{}')\n")
        fake.chmod(0o700)
        return capture_file_feed([str(fake)], root, Hub(), budget, .2, 32)

    def test_complete_raw_records_survive_display_reduction(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            stats = self.run_fake(p,2**20)
            raw = [json.loads(x) for x in (p/"raw.feed").read_text().splitlines()]
            display = [json.loads(x) for x in (p/"display.feed").read_text().splitlines()]
            self.assertEqual(sum("sig" in x for x in raw), 300)
            self.assertEqual(sum("sig" in x for x in display), 2)
            self.assertNotIn("display_sampling", raw[0]["meta"])
            self.assertEqual(display[0]["meta"]["display_sampling"]["interval_s"], .2)
            self.assertIn("end", display[-1])
            self.assertEqual(stats["coalesced_signals"], 298)
            self.assertEqual(display[-1]["end"]["policy"]["records"], 300)
            self.assertEqual(display[-1]["end"]["policy"]["unknown"], 300)

    def test_os_and_display_quotas_retain_incomplete_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            with self.assertRaisesRegex(RuntimeError,"limit|incomplete|receiver"):
                self.run_fake(p,1024)
            self.assertLessEqual((p/"raw.feed").stat().st_size, 512)
            self.assertLessEqual((p/"display.feed").stat().st_size, 512)
