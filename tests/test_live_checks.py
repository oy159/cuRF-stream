import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from run_live_checks import record_case, runtime_checks
from ui_server import Hub


class LiveChecksTests(unittest.TestCase):
    def test_physical_loss_and_skips_are_gated_separately_from_classification(self):
        case={"rx":[2],"freq":939e6}
        meta={"source":"omnicore","rx":[2],"freq":939e6,"rate":61.44e6}
        runtime={"wall_s":300.,"dropped_frames":0,"gap_frames":0,
                 "omnicore":dict.fromkeys(("host_dropped","device_overflows","transfer_errors","resyncs","discontinuities","tx_samples"),0),
                 "stage2":dict.fromkeys(("skipped_busy","skipped_ring","skipped_capacity"),0)}
        self.assertTrue(all(runtime_checks(runtime,meta,case,300.,True).values()))
        for field in runtime["omnicore"]:
            bad=copy.deepcopy(runtime)
            bad["omnicore"][field]=1
            self.assertFalse(all(runtime_checks(bad,meta,case,300.,True).values()))
        for field in runtime["stage2"]:
            bad=copy.deepcopy(runtime)
            bad["stage2"][field]=1
            self.assertFalse(runtime_checks(bad,meta,case,300.,True)["zero_stage2_skips"])
        self.assertFalse(runtime_checks(runtime,meta,case,300.,False)["complete_feed"])
        self.assertFalse(runtime_checks(runtime,{**meta,"source":"file"},case,300.,True)["physical_receiver"])
        self.assertFalse(runtime_checks(runtime,{**meta,"rx":[1]},case,300.,True)["physical_receiver"])
        self.assertFalse(runtime_checks({**runtime,"wall_s":299.},meta,case,300.,True)["duration"])

    def test_capture_limit_stops_child_and_retains_incomplete_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            fake=root/"receiver"
            meta={"meta":{"source":"omnicore","rate":61.44e6,"streams":1,"freq":939e6}}
            fake.write_text("#!"+sys.executable+"\nimport sys,time\nf=open(sys.argv[-1],'w')\nf.write("
                            +repr(json.dumps(meta)+"\n")+")\nf.flush()\ntime.sleep(30)\n")
            fake.chmod(0o700)
            case={"name":"cell900","freq":939e6,"gain":40,"threshold":6,"rx":[2]}
            with self.assertRaisesRegex(RuntimeError,"feed limit"):
                record_case(fake,root,root/"trial",case,300.,Hub(),max_bytes=8)
            self.assertTrue((root/"trial/command.json").exists())
            self.assertTrue((root/"trial/display.feed").exists())
            self.assertFalse((root/"trial/result.json").exists())


if __name__=="__main__":
    unittest.main()
