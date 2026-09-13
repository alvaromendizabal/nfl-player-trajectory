import unittest,sys,tempfile,hashlib,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import run_round

class Launcher(unittest.TestCase):
    def test_offline_exact_lock(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);kit=d/'kit';repo=d/'repo';out=d/'out'
            kit.mkdir();(repo/'scripts').mkdir(parents=True);out.mkdir()
            raw=b'version = 1\n';blob=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
            (repo/'scripts/motion_supervision.py.lock').write_bytes(raw)
            (kit/'input_contract.json').write_text(json.dumps({'round8':{'neural_lock_git_blob':blob}}))
            (kit/'worker.py').write_text('# fixture')
            a=SimpleNamespace(stage='runtime',arm='cartesian',repo=repo,out=out,parent=d/'parent',parent_kit=d/'old',audit=d/'audit')
            with patch.object(run_round,'KIT',kit),patch.object(run_round.shutil,'which',return_value='/usr/bin/uv'):
                c=run_round.command(a)
            for v in ('--frozen','--offline','--no-project','--no-python-downloads'):self.assertIn(v,c)
            self.assertEqual((out/'runtime/worker.py.lock').read_bytes(),raw)
    def test_worker_rejects_changed_lock(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);kit=d/'kit';repo=d/'repo';kit.mkdir();(repo/'scripts').mkdir(parents=True)
            (repo/'scripts/motion_supervision.py.lock').write_text('drift')
            (kit/'input_contract.json').write_text(json.dumps({'round8':{'neural_lock_git_blob':'a'*40}}))
            a=SimpleNamespace(stage='runtime',repo=repo,out=d/'out')
            with patch.object(run_round,'KIT',kit),self.assertRaises(ValueError):run_round.command(a)

if __name__=='__main__':unittest.main()
