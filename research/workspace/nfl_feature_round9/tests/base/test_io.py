import unittest
import tempfile
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fixtures import sample
from audit_io import (atomic_json,read_json,seal_json,checkpoint_npz,read_observed,safe_file,digest)
from run_round import export_report,safe_paths,LIMITS

class IOTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.p=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def test_atomic(self):atomic_json(self.p/'a.json',{'n':1});self.assertEqual(read_json(self.p/'a.json'),{'n':1})
    def test_nonfinite_reject(self):
        with self.assertRaises(ValueError):atomic_json(self.p/'a.json',{'n':float('nan')})
    def test_seal_drift(self):
        seal_json(self.p/'a.json',{'n':1})
        with self.assertRaises(ValueError):seal_json(self.p/'a.json',{'n':2})
    def test_no_targets_unpacked(self):
        p=sample();p['y']=np.array([object()],object);np.savez(self.p/'a.npz',**p)
        self.assertNotIn('y',read_observed(self.p/'a.npz'))
    def test_eval_reject(self):
        p=sample();p['train']=np.array(False);np.savez(self.p/'a.npz',**p)
        with self.assertRaises(ValueError):read_observed(self.p/'a.npz')
    def test_checkpoint_reuse(self):
        p=self.p/'a.npz';arr={'x':np.array([1.2],np.float32)}
        checkpoint_npz(p,arr);mt=p.stat().st_mtime_ns;h=digest(p);checkpoint_npz(p,arr)
        self.assertEqual(mt,p.stat().st_mtime_ns);self.assertEqual(h,digest(p))
    def test_checkpoint_value_drift(self):
        p=self.p/'a.npz';checkpoint_npz(p,{'x':np.array([1.],np.float32)})
        with self.assertRaises(ValueError):checkpoint_npz(p,{'x':np.array([1.1],np.float32)})
    def test_checkpoint_dtype_drift(self):
        p=self.p/'a.npz';checkpoint_npz(p,{'x':np.array([1.],np.float32)})
        with self.assertRaises(ValueError):checkpoint_npz(p,{'x':np.array([1.],np.float64)})
    def test_unsafe_path(self):
        with self.assertRaises(ValueError):safe_file(self.p,'../private')
    def test_symlink(self):
        (self.p/'a').write_text('x');(self.p/'b').symlink_to(self.p/'a')
        with self.assertRaises(ValueError):safe_file(self.p,'b')
    def test_no_training_command(self):self.assertNotIn('train',LIMITS)
    def test_report_private_exclusion(self):
        atomic_json(self.p/'preflight.json',{'ok':True});(self.p/'weights.pt').write_bytes(b'private')
        with zipfile.ZipFile(export_report(self.p)) as z:
            self.assertEqual(set(z.namelist()),{'preflight.json','CONTENTS.txt'})
    def test_output_overlap(self):
        a=SimpleNamespace(out=self.p,parent=self.p,parent_kit=self.p/'k',repo=self.p/'r')
        with self.assertRaises(ValueError):safe_paths(a)

if __name__=='__main__':unittest.main()

class AggregationTests(unittest.TestCase):
    def test_json_roundtrip_order_does_not_change_result(self):
        import json
        from audit_worker import combine
        changes={name:{'sum_squared_change':1.,'coordinate_count':2,'max_absolute_change':1.} for name in ['z','a']}
        arm={'changes':changes,'gradient_energy':[1.]*8,'valid_counts':[2]*8,
             'channel_zeroing':[{'sum_squared_change':1.,'coordinate_count':2} for _ in range(8)]}
        r={'input_contrast':{'sum_squared_difference':[1.]*8,'valid_counts':[2]*8,'different_counts':[1]*8},
           'arms':{'terminal':arm,'history':arm}}
        self.assertEqual(combine([r],[str(i) for i in range(8)]),combine([json.loads(json.dumps(r,sort_keys=True))],[str(i) for i in range(8)]))
