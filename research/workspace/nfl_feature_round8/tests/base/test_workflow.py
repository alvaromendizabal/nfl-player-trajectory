import json,sys,tempfile,unittest,zipfile
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from parent_support import safe_file,seal_json,atomic_json,atomic_npz,read_npz,digest
from run_round import export_report,safe_paths,LIMITS,REPORT_NAMES

class Workflow(unittest.TestCase):
    def test_path_traversal(self):
        with self.assertRaises(ValueError):safe_file(Path('/tmp'),'../abc')
    def test_absolute_path(self):
        with self.assertRaises(ValueError):safe_file(Path('/tmp'),'/abc')
    def test_symlink(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'a').write_text('x');(p/'b').symlink_to(p/'a')
            with self.assertRaises(ValueError):safe_file(p,'b')
    def test_sealed_plan_reuse(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'p.json';seal_json(p,{'a':1});h=digest(p);mt=p.stat().st_mtime_ns;seal_json(p,{'a':1});self.assertEqual(h,digest(p));self.assertEqual(mt,p.stat().st_mtime_ns)
    def test_plan_change_stops(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'p.json';seal_json(p,{'a':1})
            with self.assertRaises(ValueError):seal_json(p,{'a':2})
    def test_strict_json(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):atomic_json(Path(t)/'p.json',{'a':float('nan')})
    def test_npz_replay(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'p.npz';a=np.arange(12).reshape(4,3);atomic_npz(p,a=a);np.testing.assert_array_equal(read_npz(p)['a'],a)
    def test_report_private_exclusion(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);atomic_json(p/'summary.json',{'x':1});(p/'private.csv').write_text('not public');(p/'weights.pt').write_text('not public');atomic_npz(p/'model_input.npz',y=np.ones(3))
            z=export_report(p)
            with zipfile.ZipFile(z) as f:self.assertEqual(set(f.namelist()),{'summary.json','CONTENTS.txt'})
    def test_report_symlink_exclusion(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'private.json').write_text('{}');(p/'summary.json').symlink_to(p/'private.json')
            with zipfile.ZipFile(export_report(p)) as f:self.assertNotIn('summary.json',f.namelist())
    def test_budget_limits(self):self.assertEqual(LIMITS['train'],360);self.assertEqual(LIMITS['prepare'],360)
    def test_report_does_not_include_model_blobs(self):self.assertTrue(all(not n.endswith(('.pt','.npz','.csv')) for n in REPORT_NAMES))
    def test_outside_parent_requirement(self):
        a=SimpleNamespace(out=Path('/tmp/test/inside'),round7=Path('/tmp/test'),round5=Path('/tmp/r5'),round5_kit=Path('/tmp/k5'),round4=Path('/tmp/r4'),round4_kit=Path('/tmp/k4'),round3=Path('/tmp/r3'),round3_kit=Path('/tmp/k3'),repo=Path('/tmp/repo'))
        with self.assertRaises(ValueError):safe_paths(a)
if __name__=='__main__':unittest.main()
