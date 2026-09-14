from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import study
import run_round as runner
from origin_features import fit_ridge, predict_ridge, rmse
from fixtures import parent_fixture


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=tempfile.TemporaryDirectory()
        cls.original_parent,cls.original_repo,cls.contract=parent_fixture(Path(cls.base.name))
    @classmethod
    def tearDownClass(cls):cls.base.cleanup()
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.parent=self.root/'parent';self.repo=self.root/'repo';self.out=self.root/'out';self.out.mkdir()
        shutil.copytree(self.original_parent,self.parent);shutil.copytree(self.original_repo,self.repo)
        self.args=argparse.Namespace(repo=self.repo,parent=self.parent,out=self.out)
    def tearDown(self):self.tmp.cleanup()
    def load(self):return study.load_parent(self.parent,self.contract)
    def test_parent_replays_without_fit(self):
        with patch.object(study,'fit_ridge',side_effect=AssertionError('No parent refit allowed')):
            parent=self.load()
        self.assertEqual(parent['parent_models_replayed'],6);self.assertEqual(parent['parent_models_refitted'],0)
    def test_global_indices_skip_augmented_rows(self):
        parent=self.load();glob=parent['data']['global_indices']
        self.assertTrue((np.diff(glob)>1).any())
        for fold in parent['folds']:
            self.assertTrue(np.isin(parent['data']['keys'][fold['train'],0],fold['train_games']).all())
    def test_parent_bytes_unchanged_by_load(self):
        before={p.relative_to(self.parent).as_posix():(study.digest(p),p.stat().st_mtime_ns) for p in self.parent.rglob('*') if p.is_file()}
        self.load()
        after={p.relative_to(self.parent).as_posix():(study.digest(p),p.stat().st_mtime_ns) for p in self.parent.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
    def test_parent_manifest_tamper_stops(self):
        p=self.parent/'research/dataset_manifest.json';p.write_text(p.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'manifest differs'):self.load()
    def test_parent_feature_tamper_stops(self):
        p=next((self.parent/'research/plays').glob('*o0.npz'));p.write_bytes(p.read_bytes()+b'tamper')
        with self.assertRaisesRegex(ValueError,'checksum failed'):self.load()
    def test_parent_model_tamper_stops(self):
        p=self.parent/'research/screen/fold_1_control.npz';p.write_bytes(p.read_bytes()+b'tamper')
        with self.assertRaisesRegex(ValueError,'artifact differs'):self.load()
    def test_parent_summary_tamper_stops(self):
        p=self.parent/'research/screen_summary.json';p.write_text(p.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'screen report changed'):self.load()
    def test_safe_paths_reject_traversal(self):
        for name in ('../secret','/absolute','a\\b','C:secret'):
            with self.assertRaises(ValueError):study.safe_file(self.parent,name)
    def test_safe_paths_reject_symlink(self):
        (self.root/'link').symlink_to(self.parent,target_is_directory=True)
        with self.assertRaises(ValueError):study.safe_file(self.root,'link/research/screen_summary.json')
    def test_attribution_layout(self):
        arms=study.attribution_arms();self.assertEqual(len(arms),6)
        for n,a in arms.items():self.assertEqual(len(a['columns']),80 if n.startswith('only') else 88)
    def test_motion_layout(self):
        arms=study.motion_arms();self.assertEqual(len(arms),3)
        self.assertEqual([len(a['columns']) for a in arms.values()],[120,120,108])
    def test_train_only_scaling(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(100,4));y=rng.normal(size=(100,2))
        model=fit_ridge(x[:60],y[:60]);np.testing.assert_allclose(model['mean'],x[:60].mean(0))
        x[60:]*=1000;again=fit_ridge(x[:60],y[:60]);np.testing.assert_array_equal(model['coef'],again['coef'])
    def test_rmse_not_euclidean_distance(self):
        self.assertAlmostEqual(rmse(np.array([[0.,0.]]),np.array([[3.,4.]])),np.sqrt(12.5))
    def test_bootstrap_identical_arms_zero(self):
        y=np.arange(40).reshape(20,2).astype(float);g=np.repeat(np.arange(4),5)
        result=study.paired_interval(y,y+1,y+1,g)
        self.assertEqual(result['delta_rmse'],0);self.assertEqual(result['simultaneous_high'],0)
    def test_bootstrap_sign_and_adjustment(self):
        y=np.zeros((20,2));g=np.repeat(np.arange(4),5)
        a=np.repeat(np.arange(1,5),5)[:,None]*np.ones((20,2));b=a*.9
        result=study.paired_interval(y,a,b,g)
        self.assertLess(result['ci95_high'],0)
        self.assertLessEqual(result['simultaneous_low'],result['ci95_low'])
        self.assertGreaterEqual(result['simultaneous_high'],result['ci95_high'])
    def test_seal_rejects_protocol_drift(self):
        p=self.out/'plan.json';study.seal_json(p,{'a':1})
        with self.assertRaises(ValueError):study.seal_json(p,{'a':2})
    def test_attribution_checkpoint_reuse_and_replay(self):
        parent=self.load();one=study.screen(parent,self.out,'attribution','synthetic')
        self.assertEqual(one['new_fits_this_invocation'],18)
        paths=list((self.out/'attribution').glob('*.npz'));before={p.name:(study.digest(p),p.stat().st_mtime_ns) for p in paths}
        with patch.object(study,'fit_ridge',side_effect=AssertionError('No repeated fit')):
            two=study.screen(parent,self.out,'attribution','synthetic',replay_only=True)
        self.assertEqual(two['new_fits_this_invocation'],0);self.assertEqual(two['reused_new_arm_fits'],18)
        self.assertEqual(before,{p.name:(study.digest(p),p.stat().st_mtime_ns) for p in paths})
    def test_candidate_interrupted_receipt_recovers_without_fit(self):
        parent=self.load();data=parent['data'];f=parent['folds'][0];cols=np.arange(72)
        p=self.out/'candidate.npz';spec={'signature':'synthetic'}
        study.candidate_fit(p,data['X'],data['y'],data['keys'],f['train'],f['eval'],cols,spec,list(study.ALL_NAMES))
        p.with_suffix('.json').unlink()
        with patch.object(study,'fit_ridge',side_effect=AssertionError('Already completed')):
            _,receipt,reuse=study.candidate_fit(p,data['X'],data['y'],data['keys'],f['train'],f['eval'],cols,spec,list(study.ALL_NAMES))
        self.assertTrue(reuse);self.assertTrue(receipt['forward_replay_exact']);self.assertTrue(p.with_suffix('.json').exists())
    def test_missing_replay_never_fits(self):
        with patch.object(study,'fit_ridge',side_effect=AssertionError('Must not fit')):
            with self.assertRaisesRegex(ValueError,'Replay never fits'):study.screen(self.load(),self.out,'attribution','synthetic',replay_only=True)
    def test_new_feature_raw_pipeline_reuse_and_motion_screen(self):
        parent=self.load()
        one=runner.prepare_features(self.args,parent,self.contract,'synthetic',limit=256)
        self.assertEqual(one['new_play_checkpoints'],16)
        paths=list((self.out/'features/plays').glob('*.npz'));before={p.name:(study.digest(p),p.stat().st_mtime_ns) for p in paths}
        two=runner.prepare_features(self.args,parent,self.contract,'synthetic',limit=256)
        self.assertEqual(two['new_play_checkpoints'],0)
        self.assertEqual(before,{p.name:(study.digest(p),p.stat().st_mtime_ns) for p in paths})
        extra=runner.load_extra(parent,self.out,'synthetic')
        result=study.screen(parent,self.out,'motion','synthetic',extra)
        self.assertEqual(result['new_fits_this_invocation'],9);self.assertEqual(result['parent_models_refitted'],0)
        self.assertEqual(result['evaluation_rows'],parent['parent_summary']['evaluation_rows'])
    def test_changed_raw_stops_preparation(self):
        p=self.repo/'data/raw/train/input_2023_w01.csv';p.write_text(p.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'tracking checksum changed'):runner.prepare_features(self.args,self.load(),self.contract,'synthetic',limit=32)
    def test_report_excludes_weights_and_keys(self):
        (self.out/'motion').mkdir();study.atomic_json(self.out/'motion/summary.json',{'status':'SYNTHETIC'})
        (self.out/'motion/private.npz').write_bytes(b'private');study.atomic_json(self.out/'motion/protocol.json',{'train_indices':[1,2]})
        runner.export_report(self.out)
        with zipfile.ZipFile(self.out/'nfl_feature_round2_report.zip') as z:
            self.assertEqual(set(z.namelist()),{'motion/summary.json','CONTENTS.txt'})
    def test_watchdog_stops_blocked_child(self):
        real=subprocess.Popen
        def delayed(command,**kwargs):return real([sys.executable,'-c','import time; time.sleep(5)'],**kwargs)
        with patch.object(sys,'argv',['run_round.py','report','--out',str(self.out),'--seconds','1']):
            with patch.object(runner.subprocess,'Popen',side_effect=delayed):
                with self.assertRaises(SystemExit) as exc:runner.main()
        self.assertEqual(exc.exception.code,124)
        r=json.loads((self.out/'last_command.json').read_text());self.assertEqual(r['status'],'stopped_budget');self.assertLess(r['elapsed_seconds'],3)
    def test_lock_blocks_parallel_stage(self):
        import fcntl
        with (self.out/'.round.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with patch.object(sys,'argv',['run_round.py','report','--out',str(self.out)]):
                with self.assertRaises(SystemExit) as exc:runner.main()
        self.assertEqual(exc.exception.code,2)
