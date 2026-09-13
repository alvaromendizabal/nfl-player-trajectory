import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from experiment import compare,checkpoint,matrices
from parent_support import atomic_json,atomic_npz,digest,seal_json,safe_file
from run_round import export_report

class WorkflowTests(unittest.TestCase):
    def test_pooled_rmse_not_fold_mean(self):
        y=np.zeros((6,2));a=np.ones_like(y);b=a*.8;g=np.array([1,1,2,2,3,3]);r=compare(y,a,b,g)
        self.assertAlmostEqual(r['relative_gain'],.2);self.assertTrue(r['screen_gate']);self.assertEqual(r['planned_comparison_looks'],9)
    def test_negative_gate(self):
        r=compare(np.zeros((6,2)),np.ones((6,2)),np.ones((6,2))*2,np.array([1,1,2,2,3,3]));self.assertFalse(r['screen_gate'])
    def test_zero_gain_fails(self):
        r=compare(np.zeros((6,2)),np.ones((6,2)),np.ones((6,2)),np.array([1,1,2,2,3,3]));self.assertFalse(r['screen_gate'])
    def test_bootstrap_deterministic(self):
        a=np.arange(12).reshape(6,2)+1;b=a*.9;g=np.array([1,1,2,2,3,3]);self.assertEqual(compare(a*0,a,b,g),compare(a*0,a,b,g))
    def test_zero_control_reject(self):
        with self.assertRaises(ValueError):compare(np.zeros((4,2)),np.zeros((4,2)),np.ones((4,2)),np.array([1,1,2,2]))
    def test_single_game_reject(self):
        with self.assertRaises(ValueError):compare(np.zeros((4,2)),np.ones((4,2)),np.ones((4,2)),np.ones(4))
    def test_feature_dimensions(self):
        m=matrices({'X':np.ones((8,96))},np.ones((8,48)));self.assertEqual(m['pass_axis'].shape,(8,93));self.assertEqual(m['pass_axis_history'].shape,(8,120))
    def test_no_direct_state_bundle(self):
        d={'X':np.ones((8,96)),'state':np.full((8,62),np.nan)}
        self.assertTrue(np.isfinite(matrices(d,np.ones((8,48)))['pass_axis']).all())
    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/'features/1_2.npz';atomic_npz(p,static=np.zeros((2,21)),history=np.zeros((2,27)),keys=np.array([[1,2,1,1],[1,2,1,2]]))
            atomic_json(p.with_suffix('.json'),{'signature':'s','parent_sha256':'p','sha256':digest(p)})
            z,_=checkpoint(root,{'game':1,'play':2},'s','p');self.assertEqual(z['static'].shape,(2,21))
    def test_checkpoint_drift(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/'features/1_2.npz';atomic_npz(p,static=np.zeros((2,21)),history=np.zeros((2,27)),keys=np.zeros((2,4)))
            atomic_json(p.with_suffix('.json'),{'signature':'wrong','parent_sha256':'p','sha256':digest(p)})
            with self.assertRaises(ValueError):checkpoint(root,{'game':1,'play':2},'s','p')
    def test_orphan_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/'features/1_2.npz';p.parent.mkdir();p.write_bytes(b'partial')
            with self.assertRaises(FileNotFoundError):checkpoint(root,{'game':1,'play':2},'s','p')
    def test_manifest_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'m.json';seal_json(p,{'a':1});old=p.stat().st_mtime_ns;seal_json(p,{'a':1});self.assertEqual(old,p.stat().st_mtime_ns)
            with self.assertRaises(ValueError):seal_json(p,{'a':2})
    def test_path_escape_reject(self):
        with self.assertRaises(ValueError):safe_file(Path('/tmp'),'../secret')
    def test_symlink_reject(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'x').write_text('x');(root/'l').symlink_to(root/'x')
            with self.assertRaises(ValueError):safe_file(root,'l')
    def test_nonfinite_json_reject(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):atomic_json(Path(t)/'m.json',{'x':float('nan')})
    def test_private_report_exclusion(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);atomic_json(root/'summary.json',{'status':'fixture'});atomic_json(root/'private.json',{'truth':'private'})
            (root/'model.pkl').write_text('private');p=export_report(root)
            with zipfile.ZipFile(p) as z:self.assertEqual(set(z.namelist()),{'summary.json','CONTENTS.txt'})
