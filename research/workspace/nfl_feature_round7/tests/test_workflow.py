import json, tempfile, unittest, zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from history_study import allowed_fold3,matrices,compare
from run_round import export_report,safe_paths
from parent_support import seal_json,read_npz,atomic_npz,safe_file

class WorkflowTests(unittest.TestCase):
    def test_matrix_shapes(self):
        m=matrices(np.zeros((10,96)),np.zeros((10,21)));self.assertEqual({k:v.shape[1] for k,v in m.items()},{'support_only':81,'role_response':87,'player_response':93})
    def test_support_identical(self):
        m=matrices(np.ones((10,96)),np.arange(210).reshape(10,21));np.testing.assert_array_equal(m['role_response'][:,:81],m['support_only']);np.testing.assert_array_equal(m['player_response'][:,:87],m['role_response'])
    def test_futility_both_bad(self):self.assertFalse(allowed_fold3({'metrics':{'control':1,'role_response':1.06,'player_response':1.07}}))
    def test_futility_one_bad(self):self.assertTrue(allowed_fold3({'metrics':{'control':1,'role_response':1.06,'player_response':1.02}}))
    def test_futility_threshold(self):self.assertFalse(allowed_fold3({'metrics':{'control':1,'role_response':1.05,'player_response':1.05}}))
    def test_identity_interval(self):
        y=np.ones((30,2));p=np.zeros_like(y);r=compare(y,p,p,np.repeat([1,2,3],10));self.assertEqual(r['delta_rmse'],0);self.assertFalse(r['screen_gate'])
    def test_known_improvement(self):
        y=np.ones((30,2));p=np.zeros_like(y);r=compare(y,p,p+.5,np.repeat([1,2,3],10));self.assertEqual(r['relative_gain'],.5);self.assertTrue(r['screen_gate']);self.assertEqual(r['planned_comparison_looks'],18)
    def test_sealed_plan_drift(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'plan.json';seal_json(p,{'x':1})
            with self.assertRaises(ValueError):seal_json(p,{'x':2})
    def test_npz_atomic(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.npz';atomic_npz(p,x=np.arange(8));np.testing.assert_array_equal(read_npz(p)['x'],np.arange(8))
    def test_unsafe_path(self):
        with self.assertRaises(ValueError):safe_file(Path('/tmp'),'../secret')
    def test_symlink_path(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'a').write_text('x');(p/'b').symlink_to(p/'a')
            with self.assertRaises(ValueError):safe_file(p,'b')
    def test_report_excludes_private(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'summary.json').write_text('{"status":"synthetic"}');(p/'tables').mkdir();(p/'tables/fold_2.json').write_text('{"private":true}')
            (p/'weights.pkl').write_bytes(b'private');f=export_report(p)
            with zipfile.ZipFile(f) as z:self.assertEqual(set(z.namelist()),{'summary.json','CONTENTS.txt'})
    def test_result_root_collision(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);args=SimpleNamespace(out=p/'repo/results',round6=p/'r6',round5=p/'r5',round5_kit=p/'k5',round4=p/'r4',round4_kit=p/'k4',round3=p/'r3',round3_kit=p/'k3',repo=p/'repo')
            with self.assertRaises(ValueError):safe_paths(args)

if __name__=='__main__':unittest.main()
