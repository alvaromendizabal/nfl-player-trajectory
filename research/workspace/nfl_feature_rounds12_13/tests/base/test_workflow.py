import copy,json,sys,tempfile,unittest,zipfile,hashlib
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import audit_io as io
import run_round as run
import bridge

class Workflow(unittest.TestCase):
    def test_report_private_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);io.atomic_json(p/'preflight.json',{'status':'passed'});(p/'secret.npz').write_text('private')
            for r in (12,13):
                with zipfile.ZipFile(run.report(p,r)) as z:self.assertEqual(set(z.namelist()),{'preflight.json','CONTENTS.txt'})
    def test_report_different_names(self):
        with tempfile.TemporaryDirectory() as d:self.assertNotEqual(run.report(Path(d),12),run.report(Path(d),13))
    def test_npz_exact_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'f.npz';a={'a':np.ones(2,np.float32)};h=io.checkpoint_npz(p,a);t=p.stat().st_mtime_ns
            self.assertEqual(h,io.checkpoint_npz(p,a));self.assertEqual(t,p.stat().st_mtime_ns)
    def test_npz_float64_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'f.npz';io.checkpoint_npz(p,{'a':np.ones(2,np.float32)})
            with self.assertRaises(ValueError):io.checkpoint_npz(p,{'a':np.ones(2,np.float64)})
    def test_changed_value_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'f.npz';io.checkpoint_npz(p,{'a':np.ones(2)})
            with self.assertRaises(ValueError):io.checkpoint_npz(p,{'a':np.zeros(2)})
    def test_sealed_json_rejects_drift(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';io.seal_json(p,{'a':1})
            with self.assertRaises(ValueError):io.seal_json(p,{'a':2})
    def test_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):io.atomic_json(Path(d)/'a.json',{'a':float('nan')})
    def test_unsafe_paths(self):
        for p in ('../x','/tmp/x','a\\b','C:x'):
            with self.assertRaises(ValueError):io.safe_file(Path('/tmp'),p)
    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'x').write_text('x');(p/'link').symlink_to(p/'x')
            with self.assertRaises(ValueError):io.safe_file(p,'link')
    def test_parent_mutation_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_text('x');v={'protected':{str(p):io.digest(p)}};bridge.unchanged(v);p.write_text('z')
            with self.assertRaises(ValueError):bridge.unchanged(v)
    def test_source_hash_sensitive(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'input_contract.json').write_text('{}');(p/'a.py').write_text('x=1');a=bridge.code_signature(p);(p/'a.py').write_text('x=2');self.assertNotEqual(a,bridge.code_signature(p))
    def test_eval_label_guard(self):
        with self.assertRaisesRegex(ValueError,'Evaluation targets'):bridge.observed(SimpleNamespace(),{'train':False},{},labels=True)
    def test_round_locks_and_no_online_flag(self):
        s=Path(run.__file__).read_text();self.assertIn("(a.audit,a.parent,a.tensors)",s);self.assertNotIn("add_argument('--online'",s)
    def test_model_caps_equal(self):self.assertEqual(run.LIMITS['train'],360);self.assertEqual(run.LIMITS['profile'],180)
    def test_local_commands_no_install(self):
        a=SimpleNamespace(stage='smoke',round_no=12,arm='mask',out=Path('/tmp/a'),audit=Path('/tmp/b'),audit_kit=Path('/tmp/c'),parent=Path('/tmp/d'),parent_kit=Path('/tmp/e'),tensors=Path('/tmp/f'),repo=Path('/tmp/g'))
        cmd=run.command(a);self.assertIn('--round',cmd);self.assertNotIn('pip',cmd);self.assertNotIn('uv',cmd)
    def test_checkpoint_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'f.npz';io.checkpoint_npz(p,{'a':np.ones(2)});p.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):io.checkpoint_npz(p,{'a':np.ones(2)})
