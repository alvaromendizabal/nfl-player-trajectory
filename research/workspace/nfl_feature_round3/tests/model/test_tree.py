import json,tempfile,unittest,subprocess,sys,os
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from tree_worker import fit_axis,SETTINGS,paired

class TreeTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.out=Path(self.t.name);rng=np.random.default_rng(45)
        self.x=rng.normal(size=(180,6));self.y=self.x[:,0]**2-self.x[:,1];self.spec={**SETTINGS,'max_iter':60,'min_samples_leaf':10}
    def tearDown(self):self.t.cleanup()
    def fit(self,**kw):return fit_axis(self.out,self.x,self.y,self.x[:40],'test',settings=self.spec,**kw)
    def test_fit_finite(self):self.assertTrue(np.isfinite(self.fit()[0]).all())
    def test_replay_no_refit(self):
        pred,_=self.fit();pred2,r=self.fit(replay_only=True);np.testing.assert_array_equal(pred,pred2);self.assertEqual(r['new_chunks'],0)
    def test_continuation_matches_clean(self):
        self.fit(max_chunks=1);p,_=self.fit();clean=HistGradientBoostingRegressor(**self.spec).fit(self.x,self.y).predict(self.x[:40]);np.testing.assert_array_equal(p,clean)
    def test_fresh_process_continuation(self):
        self.fit(max_chunks=1)
        code = """import sys,numpy as np
from pathlib import Path
from tree_worker import fit_axis,SETTINGS
rng=np.random.default_rng(45);x=rng.normal(size=(180,6));y=x[:,0]**2-x[:,1]
p,r=fit_axis(Path(sys.argv[1]),x,y,x[:40],'test',settings={**SETTINGS,'max_iter':60,'min_samples_leaf':10})
np.save(Path(sys.argv[1])/'fresh.npy',p)
"""
        env=os.environ.copy();env.update({'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        subprocess.run([sys.executable,'-c',code,str(self.out)],check=True,env=env,timeout=20)
        p=np.load(self.out/'fresh.npy');clean=HistGradientBoostingRegressor(**self.spec).fit(self.x,self.y).predict(self.x[:40])
        np.testing.assert_array_equal(p,clean)
    def test_missing_replay_stops(self):
        with self.assertRaises(ValueError):self.fit(replay_only=True)
    def test_changed_source_stops(self):
        self.fit()
        with self.assertRaises(ValueError):fit_axis(self.out,self.x,self.y,self.x[:40],'changed',settings=self.spec)
    def test_corrupt_checkpoint_stops(self):
        self.fit();r=json.loads((self.out/'checkpoint.json').read_text());(self.out/r['blob']).write_bytes(b'broken')
        with self.assertRaises(ValueError):self.fit(replay_only=True)
    def test_paired_identity(self):
        y=np.ones((60,2));a=np.zeros_like(y);r=paired(y,a,a,np.repeat([1,2,3],20),repeats=100)
        self.assertEqual(r['delta_rmse'],0);self.assertFalse(r['screen_gate'])
    def test_paired_improvement(self):
        y=np.ones((60,2));a=np.zeros_like(y);r=paired(y,a,np.full_like(y,.5),np.repeat([1,2,3],20),repeats=100)
        self.assertAlmostEqual(r['relative_gain'],.5);self.assertTrue(r['screen_gate'])
    def test_invalid_matrix_rejected(self):
        self.x[0,0]=np.nan
        with self.assertRaises(ValueError):self.fit()
if __name__=='__main__':unittest.main()
