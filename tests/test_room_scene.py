from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"model"))
from make_room_scene import epochs, map_emissions


class RoomSceneTests(unittest.TestCase):
    def test_random_phase_emissions_are_clipped_at_each_epoch_boundary(self):
        e={"a":2,"b":5,"fc":1e6,"bw":1e3}
        mapped=map_emissions([e],10,3,100,120,fs=1)
        self.assertEqual([(r["t0"],r["t1"]) for r in mapped],[(100,102),(109,112),(119,120)])
        self.assertTrue(all(r["fc"]==1e6 for r in mapped))

    def test_epochs_vary_with_seed_are_bounded_and_leave_gaps(self):
        a=epochs(.8,np.random.default_rng(1))
        b=epochs(.8,np.random.default_rng(2))
        self.assertNotEqual(a,b)
        self.assertTrue(all(0<=lo<hi<=.8 for lo,hi in a+b))
        self.assertTrue(all(a[i][1]<a[i+1][0] for i in range(len(a)-1)))

    def test_continuous_carrier_retains_long_observation(self):
        for seed in range(20):
            es=epochs(.8,np.random.default_rng(seed),continuous=True)
            self.assertGreater(sum(b-a for a,b in es),.5)


if __name__=="__main__":
    unittest.main()
