import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from live_feed import DisplayReducer, capture_file_feed


def signal(t, stream=0, track=1, p=.95):
    return {"sig":{"s":stream,"id":track,"t":t,"probs":[p,1-p]}}


class DisplayReducerTests(unittest.TestCase):
    def test_latest_observation_keeps_original_probabilities_and_source_order(self):
        r = DisplayReducer(.2)
        self.assertEqual(r.push(signal(0)), [])
        self.assertEqual(r.push(signal(.1,p=.97)), [])
        rows = r.push(signal(.21,p=.98)) + r.push({"end":{}})
        self.assertEqual([x["sig"]["t"] for x in rows if "sig" in x], [.1,.21])
        self.assertEqual(rows[0]["sig"]["probs"], [.97,1-.97])
        self.assertEqual(r.stats()["coalesced_signals"], 1)

    def test_receivers_are_separate_and_every_decoder_message_survives(self):
        r = DisplayReducer(.2)
        r.push(signal(0,0)); r.push(signal(.01,1))
        a,b = {"adsb":{"crc_ok":True,"message":1}}, {"adsb":{"crc_ok":True,"message":2}}
        rows = r.push(a)+r.push(b)
        self.assertEqual([x["sig"]["s"] for x in rows if "sig" in x], [0,1])
        self.assertEqual([x for x in rows if "adsb" in x], [a,b])

    def test_capacity_flushes_without_discarding_distinct_tracks(self):
        r = DisplayReducer(1,max_pending=2)
        rows = []
        for i in range(5): rows += r.push(signal(i*.01,track=i))
        rows += r.flush()
        self.assertEqual(len(rows), 5)
        self.assertEqual(r.stats()["coalesced_signals"], 0)
        self.assertLessEqual(r.stats()["peak_pending"], 2)

    def test_every_power_row_survives_and_flushes_delayed_signal_updates(self):
        r = DisplayReducer(.2)
        r.push(signal(0))
        powers = [{"t":i*.04,"p":[i]} for i in range(10)]
        rows = []
        for p in powers: rows += r.push(p)
        self.assertEqual([x for x in rows if "p" in x], powers)
        self.assertEqual([x for x in rows if "sig" in x], [signal(0)])


@unittest.skipUnless(shutil.which("prlimit"), "Linux file quota integration")
class FileFeedTests(unittest.TestCase):
    def run_fake(self, root, budget):
        fake = root/"receiver"
        fake.write_text("#!"+sys.executable+"\nimport json,sys\nf=open(sys.argv[-1],'w')\n"
                        "f.write(json.dumps({'meta':{'source':'omnicore'}})+'\\n')\n"
                        "for i in range(300): f.write(json.dumps({'sig':{'s':0,'id':1,'t':i/1000,'probs':[.9,.1]}})+'\\n')\n"
                        "f.write(json.dumps({'end':{}})+'\\n')\nf.close()\nprint('{}')\n")
        fake.chmod(0o700)
        class Hub:
            def publish(self, line): return line
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

    def test_os_and_display_quotas_retain_incomplete_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            with self.assertRaisesRegex(RuntimeError,"limit|incomplete|receiver"):
                self.run_fake(p,1024)
            self.assertLessEqual((p/"raw.feed").stat().st_size, 512)
            self.assertLessEqual((p/"display.feed").stat().st_size, 512)
