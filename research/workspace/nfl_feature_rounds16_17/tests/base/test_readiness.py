import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from readiness import status,require_review_release
from audit_io import checkpoint_npz,digest,safe_file
import numpy as np

class Readiness(unittest.TestCase):
    def test_no_traceback_for_pending_review(self):
        with patch('readiness.require_review_release',side_effect=FileNotFoundError('missing')):
            r=status(SimpleNamespace());self.assertEqual(r['status'],'review_pending');self.assertFalse(r['ready'])
    def test_invalid_review_not_enabled(self):
        with patch('readiness.require_review_release',side_effect=ValueError('wrong signature')):
            self.assertFalse(status(SimpleNamespace())['ready'])
    def test_success_status(self):
        with patch('readiness.require_review_release',return_value={'round':16}):
            self.assertTrue(status(SimpleNamespace())['ready'])
    def test_missing_release_does_not_approve(self):
        with tempfile.TemporaryDirectory() as d:
            a=SimpleNamespace(out=Path(d),round_no=16)
            with patch('readiness.reviewed_readiness',return_value={'source_signature':'x','readiness_hash':'h'}):
                with self.assertRaises(ValueError):require_review_release(a)
    def test_partial_release_does_not_approve(self):
        with tempfile.TemporaryDirectory() as d:
            a=SimpleNamespace(out=Path(d),round_no=16);(a.out/'review_release.json').write_text('{}')
            with patch('readiness.reviewed_readiness',return_value={'source_signature':'x','readiness_hash':'h'}):
                with self.assertRaises(ValueError):require_review_release(a)
    def test_checkpoint_exact_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.npz';x={'values':np.ones((2,3),np.float32),'valid':np.ones((2,3),bool)}
            h=checkpoint_npz(p,x);checkpoint_npz(p,x);self.assertEqual(digest(p),h)
    def test_checkpoint_dtype_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.npz';checkpoint_npz(p,{'x':np.ones(2,np.float32)})
            with self.assertRaises(ValueError):checkpoint_npz(p,{'x':np.ones(2,np.float64)})
    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):safe_file(Path('/tmp'),'../x')
if __name__=='__main__':unittest.main()
