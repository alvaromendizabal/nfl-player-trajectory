import hashlib,json,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
from data_pipeline import choose_plays,verify_shard
from parent_support import atomic_json,atomic_npz,digest,safe_file,seal_json
from run_round import export_report

class WorkflowTests(unittest.TestCase):
    def test_selection_deterministic(self):
        candidates=[(g,p) for g in range(10) for p in range(10)];old=candidates[:10]
        self.assertEqual(choose_plays(candidates,old,50),choose_plays(candidates[::-1],old,50))
    def test_selection_keeps_original(self):
        c=[(g,p) for g in range(10) for p in range(10)];self.assertTrue(set(c[:10]).issubset(choose_plays(c,c[:10],40)))
    def test_selection_unique(self):
        c=[(g,p) for g in range(10) for p in range(10)];self.assertEqual(len(set(choose_plays(c,c[:10],40))),40)
    def test_selection_missing_original(self):
        with self.assertRaises(ValueError):choose_plays([(1,2)],[(2,3)],1)
    def test_insufficient_plays(self):
        with self.assertRaises(ValueError):choose_plays([(1,2)],[(1,2)],4)
    def test_smaller_than_parent_reject(self):
        with self.assertRaises(ValueError):choose_plays([(1,2),(2,3)],[(1,2),(2,3)],1)
    def test_atomic_readback(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.json';atomic_json(p,{'x':3});self.assertEqual(json.loads(p.read_text()),{'x':3})
    def test_plan_drift_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.json';seal_json(p,{'x':3})
            with self.assertRaises(ValueError):seal_json(p,{'x':4})
    def test_unsafe_path_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):safe_file(Path(t),'../out')
    def test_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):atomic_json(Path(t)/'a.json',{'x':float('nan')})
    def test_report_excludes_private(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);(out/'private.pkl').write_bytes(b'private');atomic_json(out/'preflight.json',{'status':'test'})
            export_report(out)
            with zipfile.ZipFile(out/'nfl_feature_round3_report.zip') as z:self.assertNotIn('private.pkl',z.namelist());self.assertIn('preflight.json',z.namelist())
    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'x').write_text('a');(p/'y').symlink_to(p/'x')
            with self.assertRaises(ValueError):safe_file(p,'y')
if __name__=='__main__':unittest.main()
