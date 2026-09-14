import unittest,tempfile,json,zipfile,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from audit_io import safe_file,seal_json,checkpoint_npz,digest
from core import arrays,load_play,game_play
from run_round import report,paths,LIMITS

class Tests(unittest.TestCase):
    def test_no_training_stage(self):self.assertNotIn('train',LIMITS)
    def test_no_online_option(self):
        p=Path(__file__).resolve().parents[2]/'run_round.py'
        r=subprocess.run([sys.executable,str(p),'audit','--online'],capture_output=True)
        self.assertNotEqual(r.returncode,0)
    def test_no_budget_increase(self):
        p=Path(__file__).resolve().parents[2]/'run_round.py'
        r=subprocess.run([sys.executable,str(p),'audit','--seconds','999'],capture_output=True)
        self.assertNotEqual(r.returncode,0)
    def test_unsafe_path(self):
        with self.assertRaises(ValueError):safe_file(Path('/tmp'),'../x')
    def test_absolute_path(self):
        with self.assertRaises(ValueError):safe_file(Path('/tmp'),'/x')
    def test_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'file').write_text('a');(p/'link').symlink_to(p/'file')
            with self.assertRaises(ValueError):safe_file(p,'link')
    def test_seal_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s.json';seal_json(p,{'a':1});old=digest(p)
            with self.assertRaises(ValueError):seal_json(p,{'a':2})
            self.assertEqual(digest(p),old)
    def test_checkpoint_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.npz';x={'x':np.arange(5)};sha=checkpoint_npz(p,x);t=p.stat().st_mtime_ns
            self.assertEqual(sha,checkpoint_npz(p,x));self.assertEqual(t,p.stat().st_mtime_ns)
    def test_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.npz';checkpoint_npz(p,{'x':np.ones(4)})
            with self.assertRaises(ValueError):checkpoint_npz(p,{'x':np.zeros(4)})
    def test_fields_not_unpacked(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.npz';np.savez(p,x=np.arange(2),y=np.array([object()],dtype=object))
            np.testing.assert_array_equal(arrays(p,['x'])['x'],np.arange(2))
    def test_evaluation_rejected_before_access(self):
        with self.assertRaises(ValueError):load_play(None,{'train':False},None)
    def test_report_excludes_private(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'training_audit.json').write_text('{"status":"test"}');(p/'secret.npz').write_bytes(b'x')
            (p/'coverage_private.json').write_text('{"game_id":123}')
            with zipfile.ZipFile(report(p)) as z:
                self.assertEqual(set(z.namelist()),{'training_audit.json','CONTENTS.txt'})
    def test_parse_game(self):self.assertEqual(game_play({'parent_path':'plays/2023091000_2.npz'}),(2023091000,2))
    def test_bad_game(self):
        with self.assertRaises(ValueError):game_play({'parent_path':'plays/a_2.npz'})
    def test_output_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            with self.assertRaises(ValueError):paths(SimpleNamespace(out=p/'parent/sub',parent=p/'parent',parent_kit=p/'kit',tensors=p/'tensors',repo=p/'repo'))
    def test_nonfinite_seal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):seal_json(Path(d)/'x.json',{'x':float('nan')})
