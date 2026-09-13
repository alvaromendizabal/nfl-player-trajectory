import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import zipfile
import numpy as np
from confirmation import (validate_folds, matrices, training_mask, game_statistics, compare,
                          influence, slice_statistics, may_continue_fold3, ARMS, verified_fold_report)
from parent_support import atomic_json, seal_json, hash_json, safe_file, digest
from run_round import export_report, safe_paths, LIMITS


def contract():
    return {'bootstrap_resamples': 1000, 'nominal_total_comparison_looks': 10,
            'minimum_relative_gain': .01, 'futility_relative_deterioration': .05}


def fold_fixture():
    games = np.array([2023090100,2023090200,2023090300,2023090400,2023090500,2023090600])
    keys = np.column_stack([games, np.ones((6,3),dtype=int)])
    folds = [{'fold':i+1,'train_games':games[:i+3].tolist(),'validation_games':[int(games[i+3])]} for i in range(3)]
    return {'keys':keys},folds


class ConfirmationTests(unittest.TestCase):
    def test_fold_report_receipt_verified(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);atomic_json(p/'fold_2/summary.json',{'experiment_signature':'s','metric':.5})
            atomic_json(p/'fold_2/summary.receipt.json',{'experiment_signature':'s','sha256':digest(p/'fold_2/summary.json')})
            self.assertEqual(verified_fold_report(p,2)['metric'],.5)
    def test_fold_report_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);atomic_json(p/'fold_2/summary.json',{'experiment_signature':'s','metric':.5})
            atomic_json(p/'fold_2/summary.receipt.json',{'experiment_signature':'s','sha256':digest(p/'fold_2/summary.json')})
            atomic_json(p/'fold_2/summary.json',{'experiment_signature':'s','metric':.3})
            self.assertRaises(ValueError,verified_fold_report,p,2)
    def test_original_ordered_folds(self):
        d,f=fold_fixture();self.assertEqual([p['training_rows'] for p in validate_folds(d,f)],[3,4,5])
    def test_eval_overlap_rejected(self):
        d,f=fold_fixture();f[2]['validation_games']=f[1]['validation_games'];self.assertRaises(ValueError,validate_folds,d,f)
    def test_within_fold_overlap_rejected(self):
        d,f=fold_fixture();f[0]['validation_games']=f[0]['train_games'][:1];self.assertRaises(ValueError,validate_folds,d,f)
    def test_same_date_rejected(self):
        d,f=fold_fixture();f[0]['validation_games']=[2023090301];self.assertRaises(ValueError,validate_folds,d,f)
    def test_nonexpanding_rejected(self):
        d,f=fold_fixture();f[1]['train_games']=f[1]['train_games'][1:];self.assertRaises(ValueError,validate_folds,d,f)
    def test_missing_game_rejected(self):
        d,f=fold_fixture();f[2]['validation_games']=[2023100100];self.assertRaises(ValueError,validate_folds,d,f)
    def test_duplicate_declared_games(self):
        d,f=fold_fixture();f[0]['train_games']*=2;self.assertRaises(ValueError,validate_folds,d,f)
    def test_wrong_fold_numbers(self):
        d,f=fold_fixture();f[0]['fold']=2;self.assertRaises(ValueError,validate_folds,d,f)
    def test_matrix_widths_no_goal(self):
        d={'X':np.zeros((5,96)),'state':np.ones((5,62)),'goal':np.full((5,26),np.nan)}
        m=matrices(d);self.assertEqual({k:v.shape[1] for k,v in m.items()},{'control':72,'direct_state':134})
        self.assertTrue(all(np.isfinite(x).all() for x in m.values()))
    def test_no_failed_arm(self):self.assertEqual(ARMS,('control','direct_state'))
    def test_validation_does_not_screen_features(self):
        x=np.array([[0.,3.],[1.,3.],[1.,9.]])
        self.assertEqual(training_mask(x,np.array([0,1])).tolist(),[True,False])
    def test_empty_training_stops(self):self.assertRaises(ValueError,training_mask,np.ones((2,3)),np.array([],int))
    def test_constant_training_stops(self):self.assertRaises(ValueError,training_mask,np.ones((2,3)),np.array([0,1]))
    def test_game_weighting_official_formula(self):
        y=np.zeros((4,2));a=np.array([[2.,0.],[2.,0.],[2.,0.],[10.,0.]])
        s=game_statistics(y,a,a*.8,np.array([1,1,1,2]));r=compare(s,contract())
        self.assertAlmostEqual(r['control_rmse'],np.sqrt(112/8));self.assertAlmostEqual(r['relative_gain'],.2)
    def test_repeat_deterministic(self):
        y=np.zeros((4,2));a=np.ones((4,2));s=game_statistics(y,a,a*.8,np.arange(4))
        self.assertEqual(compare(s,contract()),compare(s,contract()))
    def test_sign_good_and_bad(self):
        y=np.zeros((5,2));a=np.ones((5,2));g=np.arange(5)
        good=compare(game_statistics(y,a,a*.7,g),contract());bad=compare(game_statistics(y,a,a*1.1,g),contract())
        self.assertTrue(good['numerical_screen_passed']);self.assertFalse(bad['numerical_screen_passed'])
        self.assertLess(good['adjusted_high'],0);self.assertGreater(bad['adjusted_low'],0)
    def test_one_game_bootstrap_rejected(self):
        y=np.zeros((2,2));s=game_statistics(y,y+1,y+.9,np.ones(2));self.assertRaises(ValueError,compare,s,contract())
    def test_zero_control_rejected(self):
        y=np.zeros((2,2));s=game_statistics(y,y,y+.9,np.arange(2));self.assertRaises(ValueError,compare,s,contract())
    def test_nonfinite_predictions_rejected(self):
        y=np.zeros((2,2));a=y.copy();a[0,0]=np.nan;self.assertRaises(ValueError,game_statistics,y,a,y,np.arange(2))
    def test_wrong_prediction_shape(self):self.assertRaises(ValueError,game_statistics,np.zeros((2,2)),np.zeros((2,1)),np.zeros((2,2)),np.arange(2))
    def test_influence_uniform_improvement(self):
        y=np.zeros((6,2));a=np.ones((6,2));r=influence(game_statistics(y,a,a*.9,np.arange(6)))
        self.assertAlmostEqual(r['delete_one_game_min_gain'],.1);self.assertEqual(r['delete_one_game_nonpositive_count'],0)
    def test_slice_sums_reproduce_error(self):
        y=np.zeros((3,2));a=np.ones((3,2));keys=np.array([[1,1,1,1],[1,1,1,10],[2,1,1,11]])
        r=slice_statistics(y,{'control':a},keys,np.array([0,1,1]));h=[v for v in r if 'second' in v['slice']]
        self.assertEqual(sum(v['rows'] for v in h),3);self.assertEqual(sum(v['sse'] for v in h),6)
    def test_severe_deterioration_stops(self):self.assertFalse(may_continue_fold3({'comparison':{'relative_gain':-.051}},contract()))
    def test_exact_futility_boundary(self):self.assertFalse(may_continue_fold3({'comparison':{'relative_gain':-.05}},contract()))
    def test_modest_negative_is_not_hidden(self):self.assertTrue(may_continue_fold3({'comparison':{'relative_gain':-.02}},contract()))
    def test_positive_can_continue(self):self.assertTrue(may_continue_fold3({'comparison':{'relative_gain':.02}},contract()))
    def test_no_install_stage(self):self.assertNotIn('runtime',LIMITS);self.assertNotIn('online',LIMITS)
    def test_sealed_plan_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'p.json';seal_json(p,{'a':1});before=p.stat().st_mtime_ns;seal_json(p,{'a':1})
            self.assertEqual(before,p.stat().st_mtime_ns);self.assertRaises(ValueError,seal_json,p,{'a':2})
    def test_report_excludes_arrays_models_and_keys(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);atomic_json(p/'summary.json',{'metric':.5});atomic_json(p/'fold_2/summary.json',{'metric':.5})
            (p/'model.pkl').write_bytes(b'x');(p/'private.csv').write_text('target,game_id');atomic_json(p/'feature_manifest.json',{'keys':[123]})
            with zipfile.ZipFile(export_report(p)) as z:self.assertEqual(set(z.namelist()),{'summary.json','fold_2/summary.json','CONTENTS.txt'})
    def test_report_symlink_parent_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'secret').mkdir();atomic_json(p/'secret/summary.json',{'secret':1});(p/'fold_2').symlink_to(p/'secret')
            with zipfile.ZipFile(export_report(p)) as z:self.assertNotIn('fold_2/summary.json',z.namelist())
    def test_report_invalid_json_stops(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'summary.json').write_text('broken');self.assertRaises(json.JSONDecodeError,export_report,p)
    def test_unsafe_filename_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ['../x','/x','a\\b','C:/x']:
                self.assertRaises(ValueError,safe_file,Path(td),name)
    def test_output_overlap_with_prior_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);args=SimpleNamespace(out=p/'old/results',round4=p/'old',round4_kit=p/'r4kit',round3=p/'r3',round3_kit=p/'r3kit',repo=p/'repo')
            self.assertRaises(ValueError,safe_paths,args)
    def test_hash_stable_order(self):self.assertEqual(hash_json({'a':1,'b':2}),hash_json({'b':2,'a':1}))

if __name__=='__main__':unittest.main()
