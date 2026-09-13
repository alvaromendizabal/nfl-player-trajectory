import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import zipfile
import numpy as np
from parent_support import safe_file,atomic_json,atomic_npz,digest,seal_json,hash_json
from state_features import DIRECT_NAMES,GOAL_NAMES
from study import checkpoint_read,validate_arrays
from run_round import export_report,safe_paths,LIMITS

class SafetyTests(unittest.TestCase):
    def test_no_extra_folds_or_online_stage(self):
        self.assertNotIn('fold2',LIMITS);self.assertNotIn('runtime',LIMITS)
    def test_report_does_not_export_private_files(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);atomic_json(root/'summary.json',{'metric':.7})
            (root/'private.csv').write_text('secret');(root/'model.pkl').write_bytes(b'x')
            with zipfile.ZipFile(export_report(root)) as z:
                self.assertEqual(set(z.namelist()),{'summary.json','CONTENTS.txt'})
    def test_report_symlink_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'secret').write_text('{}');(root/'summary.json').symlink_to(root/'secret')
            with zipfile.ZipFile(export_report(root)) as z:self.assertNotIn('summary.json',z.namelist())
    def test_unsafe_input_path_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            for path in ['../x','/tmp/x','x\\y','C:/x']:
                with self.assertRaises(ValueError):safe_file(Path(td),path)
    def test_sealed_protocol_change_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'seal.json';seal_json(path,{'a':1});seal_json(path,{'a':1})
            with self.assertRaises(ValueError):seal_json(path,{'a':2})
    def test_output_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);args=SimpleNamespace(out=p/'old/out',parent=p/'old',repo=p/'repo',legacy=p/'legacy')
            with self.assertRaises(ValueError):safe_paths(args)
    def test_source_hash_order_stable(self):self.assertEqual(hash_json({'a':1,'b':2}),hash_json({'b':2,'a':1}))
    def test_duplicate_keys_rejected(self):
        d={'X':np.zeros((2,96)),'y':np.zeros((2,2)),'keys':np.ones((2,4),dtype=int),'role':np.zeros(2)}
        with self.assertRaises(ValueError):validate_arrays(d,100)
    def test_checkpoint_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'f.npz';atomic_npz(p,state=np.ones((2,len(DIRECT_NAMES))),goal=np.ones((2,len(GOAL_NAMES))),keys=np.array([[1,1,1,1],[1,1,1,2]]),signature=np.array('s'))
            atomic_json(p.with_suffix('.json'),{'signature':'s','parent_sha256':'p','sha256':digest(p)})
            checkpoint_read(p,'s','p')
            with p.open('ab') as f:f.write(b'x')
            with self.assertRaises(ValueError):checkpoint_read(p,'s','p')
    def test_checkpoint_parent_drift_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'f.npz';atomic_npz(p,state=np.zeros((1,62)),goal=np.zeros((1,26)),keys=np.ones((1,4),int),signature=np.array('s'))
            atomic_json(p.with_suffix('.json'),{'signature':'s','parent_sha256':'p','sha256':digest(p)})
            with self.assertRaises(ValueError):checkpoint_read(p,'s','changed')

if __name__=='__main__':unittest.main()
