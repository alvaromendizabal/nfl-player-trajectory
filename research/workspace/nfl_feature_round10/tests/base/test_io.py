import unittest,tempfile,json,zipfile,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from audit_io import atomic_json,safe_file,seal_json,checkpoint_npz,digest
from data_io import read_arrays
from run_round import export_report,safe_paths,LIMITS

class Files(unittest.TestCase):
    def test_json(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.json';atomic_json(p,{'value':3});self.assertEqual(json.loads(p.read_text()),{'value':3})
    def test_nan_json(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):atomic_json(Path(t)/'a.json',{'x':float('nan')})
    def test_seal_drift(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.json';seal_json(p,{'v':1})
            with self.assertRaises(ValueError):seal_json(p,{'v':2})
    def test_path_escape(self):
        with self.assertRaises(ValueError):safe_file(Path('/tmp'),'../abc')
    def test_symlink(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);(d/'file').write_text('a');(d/'link').symlink_to(d/'file')
            with self.assertRaises(ValueError):safe_file(d,'link')
    def test_npz_reuse_no_write(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.npz';v={'x':np.arange(6,dtype=np.float32)};checkpoint_npz(p,v);before=(digest(p),p.stat().st_mtime_ns)
            checkpoint_npz(p,v);self.assertEqual(before,(digest(p),p.stat().st_mtime_ns))
    def test_npz_dtype_drift(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.npz';checkpoint_npz(p,{'x':np.arange(6,dtype=np.float32)})
            with self.assertRaises(ValueError):checkpoint_npz(p,{'x':np.arange(6,dtype=np.float64)})
    def test_npz_value_drift(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.npz';checkpoint_npz(p,{'x':np.arange(6)})
            with self.assertRaises(ValueError):checkpoint_npz(p,{'x':np.arange(6)+1})
    def test_read_without_target(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.npz';np.savez(p,x=np.arange(3),y=np.array([object()],dtype=object))
            np.testing.assert_equal(read_arrays(p,['x'])['x'],np.arange(3))
    def test_report_no_private(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);atomic_json(d/'summary.json',{'status':'example'})
            (d/'weights.pt').write_bytes(b'private');(d/'targets.csv').write_bytes(b'private')
            with zipfile.ZipFile(export_report(d)) as z:
                self.assertEqual(set(z.namelist()),{'summary.json','CONTENTS.txt'})
    def test_report_no_symlink(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);(d/'private').write_text('{}');(d/'summary.json').symlink_to(d/'private')
            with self.assertRaises(ValueError):export_report(d)
    def test_separate_output(self):
        a=SimpleNamespace(out=ROOT/'private',parent=ROOT,parent_kit=Path('/tmp/oldkit'),audit=Path('/tmp/audit'),repo=Path('/tmp/repo'))
        with self.assertRaises(ValueError):safe_paths(a)
    def test_hard_limits(self):self.assertEqual(LIMITS['train'],360)
    def test_no_online_stage(self):self.assertNotIn('online',LIMITS)

if __name__=='__main__':unittest.main()
