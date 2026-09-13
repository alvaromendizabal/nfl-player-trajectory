from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from origin_features import (WIDTH,ROLES,ALL_NAMES,BASE_NAMES,STATE_NAMES,build_view,query_features,
                            make_example,IneligibleOrigin,velocity,rmse,fit_ridge,predict_ridge,
                            chronological_folds,paired_bootstrap)
from run_round import choose_plays,screen_arrays,atomic_npz,digest,load_dataset


def fixture(game=2023091001,play=1,seed=1):
    rng=np.random.default_rng(seed);records=[];labels=[]
    horizon=int(rng.integers(12,32));cutoff=40
    for player in range(4):
        ident=100+player
        base=np.array([30.+player*3,15.+player*4])+rng.normal(0,.3,2)
        vel=np.array([2.8+player*.2, .3-player*.1])+rng.normal(0,.05,2)
        acceleration=np.array([.08,.015])*(player+1)
        def pos(t):return base+vel*t+.5*acceleration*t*t
        for frame in range(1,cutoff+1):
            t=(frame-cutoff)/10;xy=pos(t);v=vel+acceleration*t
            records.append(dict(game_id=game,play_id=play,nfl_id=ident,frame_id=frame,x=xy[0],y=xy[1],
                s=np.linalg.norm(v),a=np.linalg.norm(acceleration),dir=np.rad2deg(np.arctan2(v[0],v[1]))%360,
                o=90.,player_role=ROLES[player],player_side='Defense' if player==1 else 'Offense',
                play_direction='right',player_to_predict=player<2,num_frames_output=horizon,
                ball_land_x=44.,ball_land_y=19.))
        if player<2:
            for f in range(1,horizon+1):
                xy=pos(f/10);labels.append(dict(game_id=game,play_id=play,nfl_id=ident,frame_id=f,x=xy[0],y=xy[1]))
    return pd.DataFrame(records),pd.DataFrame(labels)


class FeatureTests(unittest.TestCase):
    def setUp(self):self.raw,self.out=fixture()
    def test_dimensions(self):
        e=make_example(self.raw,self.out,0)
        self.assertEqual(e['X'].shape,(len(self.out),len(ALL_NAMES)))
        self.assertEqual(len(ALL_NAMES)-len(BASE_NAMES),24)
    def test_original_label_key_preservation(self):
        e=make_example(self.raw,self.out,0)
        np.testing.assert_array_equal(e['keys'],self.out.sort_values(['nfl_id','frame_id'])[['game_id','play_id','nfl_id','frame_id']])
    def test_augmented_original_labels_preserved(self):
        e=make_example(self.raw,self.out,10)
        k=e['keys'][~e['prethrow']].copy();k[:,3]-=10
        np.testing.assert_array_equal(k,self.out.sort_values(['nfl_id','frame_id'])[['game_id','play_id','nfl_id','frame_id']])
    def test_withheld_rows_are_targets(self):
        e=make_example(self.raw,self.out,10)
        self.assertEqual(int(e['prethrow'].sum()),20)
        self.assertEqual(len(e['y']),len(self.out)+20)
    def test_ground_truth_reconstruction(self):
        e=make_example(self.raw,self.out,0)
        expected=self.out.sort_values(['nfl_id','frame_id'])[['x','y']].to_numpy()
        terminal=self.raw.sort_values('frame_id').groupby('nfl_id').tail(1).set_index('nfl_id')
        actual=e['y']+e['cv']+terminal.loc[e['keys'][:,2],['x','y']].to_numpy()
        np.testing.assert_allclose(actual,expected)
    def test_future_rows_rejected(self):
        with self.assertRaises(ValueError):build_view(self.raw,origin=30,offset=10)
    def test_withheld_mutation_cannot_change_features(self):
        altered=self.raw.copy();altered.loc[altered.frame_id>30,'x']+=20
        a=make_example(self.raw,self.out,10);b=make_example(altered,self.out,10)
        np.testing.assert_array_equal(a['X'],b['X'])
        self.assertFalse(np.array_equal(a['y'],b['y']))
    def test_label_mutation_cannot_change_features(self):
        y=self.out.copy();y['x']+=100
        np.testing.assert_array_equal(make_example(self.raw,self.out,5)['X'],make_example(self.raw,y,5)['X'])
    def test_input_immutable(self):
        r=self.raw.copy(deep=True);y=self.out.copy(deep=True);make_example(self.raw,self.out,10)
        pd.testing.assert_frame_equal(r,self.raw);pd.testing.assert_frame_equal(y,self.out)
    def test_opposite_play_direction_parity(self):
        r=self.raw.copy();y=self.out.copy()
        for d in [r,y]:d['x']=120-d.x;d['y']=WIDTH-d.y
        r['dir']=(r.dir+180)%360;r['o']=(r.o+180)%360;r['play_direction']='left'
        r['ball_land_x']=120-r.ball_land_x;r['ball_land_y']=WIDTH-r.ball_land_y
        a=make_example(self.raw,self.out,10);b=make_example(r,y,10)
        np.testing.assert_allclose(a['X'],b['X'],atol=1e-12)
        np.testing.assert_allclose(a['y'],b['y'],atol=1e-12)
    def test_lateral_reflection_target(self):
        r=self.raw.copy();y=self.out.copy();r['y']=WIDTH-r.y;y['y']=WIDTH-y.y
        r['dir']=(180-r.dir)%360;r['o']=(180-r.o)%360;r['ball_land_y']=WIDTH-r.ball_land_y
        a=make_example(self.raw,self.out,0);b=make_example(r,y,0)
        np.testing.assert_allclose(a['y'][:,0],b['y'][:,0],atol=1e-12)
        np.testing.assert_allclose(a['y'][:,1],-b['y'][:,1],atol=1e-12)
    def test_gap_not_interpolated(self):
        g=self.raw[self.raw.nfl_id==100].iloc[[0,1,4]].drop(columns=['s','dir'])
        v,ok=velocity(g,False);np.testing.assert_array_equal(ok,[False,True,False]);np.testing.assert_array_equal(v[-1],[0,0])
    def test_missing_withheld_frame_not_fabricated(self):
        r=self.raw[~((self.raw.nfl_id==100)&(self.raw.frame_id==35))]
        e=make_example(r,self.out,10);self.assertEqual(int(e['prethrow'].sum()),19)
    def test_scored_player_not_yet_observed_blocks_augmentation(self):
        r=self.raw[~((self.raw.nfl_id==100)&(self.raw.frame_id<=30))]
        with self.assertRaises(IneligibleOrigin):make_example(r,self.out,10)
    def test_partial_history_allowed(self):
        r=self.raw[self.raw.frame_id>=35];e=make_example(r,self.out,0);self.assertTrue(np.isfinite(e['X']).all())
    def test_missing_telemetry_has_mask(self):
        r=self.raw.drop(columns=['s','dir','o']);e=make_example(r,self.out,0);self.assertTrue(np.isfinite(e['X']).all())
    def test_zero_speed_finite(self):
        r=self.raw.copy();r['s']=0.;e=make_example(r,self.out,0);self.assertTrue(np.isfinite(e['X']).all())
    def test_unknown_role_rejected(self):
        r=self.raw.copy();r.loc[0,'player_role']='new'
        with self.assertRaises(ValueError):make_example(r,self.out,0)
    def test_ambiguous_anchor_rejected(self):
        r=self.raw.copy();r.loc[r.nfl_id.eq(103),'player_role']='Targeted Receiver'
        with self.assertRaises(ValueError):make_example(r,self.out,0)
    def test_duplicate_observation_rejected(self):
        with self.assertRaises(ValueError):make_example(pd.concat([self.raw,self.raw.iloc[:1]]),self.out,0)
    def test_duplicate_target_rejected(self):
        with self.assertRaises(ValueError):make_example(self.raw,pd.concat([self.out,self.out.iloc[:1]]),0)
    def test_missing_original_target_rejected(self):
        with self.assertRaises(ValueError):make_example(self.raw,self.out.iloc[1:],0)
    def test_wrong_play_labels_rejected(self):
        y=self.out.copy();y.play_id+=1
        with self.assertRaises(ValueError):make_example(self.raw,y,0)
    def test_nonfinite_target_rejected(self):
        y=self.out.copy();y.loc[0,'x']=np.nan
        with self.assertRaises(ValueError):make_example(self.raw,y,0)
    def test_metadata_change_rejected(self):
        r=self.raw.copy();r.loc[0,'ball_land_x']+=1
        with self.assertRaises(ValueError):make_example(r,self.out,0)
    def test_query_horizon_rejected(self):
        v=build_view(self.raw,origin=40)
        with self.assertRaises(ValueError):query_features(v,np.array([100]),np.array([100.]))
    def test_offset_zero_no_pseudo_labels(self):
        e=make_example(self.raw,self.out,0);self.assertFalse(e['prethrow'].any())
    def test_negative_offset_rejected(self):
        with self.assertRaises(ValueError):make_example(self.raw,self.out,-1)
    def test_large_offset_rejected(self):
        with self.assertRaises(ValueError):make_example(self.raw,self.out,21)
    def test_nfl_ids_not_features(self):
        r=self.raw.copy();y=self.out.copy();r.nfl_id+=1000;y.nfl_id+=1000
        np.testing.assert_array_equal(make_example(self.raw,self.out,0)['X'],make_example(r,y,0)['X'])
    def test_row_order_invariance(self):
        a=make_example(self.raw,self.out,5);b=make_example(self.raw.sample(frac=1,random_state=3),self.out.sample(frac=1,random_state=4),5)
        np.testing.assert_array_equal(a['X'],b['X']);np.testing.assert_array_equal(a['y'],b['y'])
    def test_stale_age_includes_forecast_origin(self):
        r=self.raw[~((self.raw.nfl_id==100)&(self.raw.frame_id>38))]
        v=build_view(r,origin=40);self.assertAlmostEqual(v.ages[100],.2)
        _,cv=query_features(v,np.array([100]),np.array([.1]));np.testing.assert_allclose(cv[0],v.velocities[100]*.3)


class ScientificTests(unittest.TestCase):
    def test_metric_coordinate_denominator(self):
        self.assertAlmostEqual(rmse(np.zeros((1,2)),np.array([[3,4]])),np.sqrt(25/2))
    def test_metric_nonfinite_rejected(self):
        with self.assertRaises(ValueError):rmse(np.zeros((1,2)),np.array([[np.nan,0]]))
    def test_scaling_not_validation_dependent(self):
        x=np.arange(200).reshape(100,2).astype(float);y=x*.5
        m=fit_ridge(x,y);b=copy.deepcopy(m);predict_ridge(m,x*1e6)
        for k in b:np.testing.assert_array_equal(m[k],b[k])
    def test_constant_features_screened_training_only(self):
        x=np.c_[np.ones(20),np.arange(20)];m=fit_ridge(x,np.c_[np.arange(20),np.arange(20)])
        np.testing.assert_array_equal(m['keep'],[False,True])
    def test_ridge_exact_serialization_replay(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(50,10));y=rng.normal(size=(50,2));m=fit_ridge(x,y)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.npz';atomic_npz(p,**m)
            with np.load(p,allow_pickle=False) as z:mm={k:z[k] for k in z.files}
            np.testing.assert_array_equal(predict_ridge(m,x),predict_ridge(mm,x))
    def test_folds_date_disjoint(self):
        games=np.array([2023090100+d*100+j for d in range(16) for j in range(2)])
        seen=[]
        for f in chronological_folds(games):
            self.assertLess(max(f['train_games']//100),min(f['validation_games']//100));seen.extend(f['validation_games'])
        self.assertEqual(len(seen),len(set(seen)))
    def test_paired_bootstrap_sign(self):
        y=np.zeros((12,2));a=np.ones_like(y);b=a*.5;g=np.repeat(np.arange(3),4)
        ci=paired_bootstrap(y,a,b,g,repeats=50);self.assertLess(ci['ci_high'],0)
    def test_selection_reproducible_and_game_diverse(self):
        keys=[(g,p) for g in range(20) for p in range(10)]
        a=choose_plays(keys,32);self.assertEqual(a,choose_plays(list(reversed(keys)),32));self.assertEqual(len(set(g for g,p in a)),20)
    def test_complete_screen_and_no_refit_resume(self):
        data={}
        for j in range(16):
            raw,out=fixture(game=2023090100+j*100,seed=j+1)
            for offset in (0,5):
                e=make_example(raw,out,offset)
                for k,v in e.items():
                    if isinstance(v,np.ndarray):data.setdefault(k,[]).append(v)
        data={k:np.concatenate(v) for k,v in data.items()}
        with tempfile.TemporaryDirectory() as d:
            result=screen_arrays(data,Path(d),repeats=30,max_rows=512,signature='synthetic')
            self.assertEqual(result['new_fits'],9);self.assertEqual(len(result['fits']),9)
            files={p.name:(digest(p),p.stat().st_mtime_ns) for p in (Path(d)/'screen').glob('fold_*.npz')}
            second=screen_arrays(data,Path(d),repeats=30,max_rows=512,signature='synthetic')
            self.assertEqual(second['new_fits'],0);self.assertEqual(second['reused_fits'],9)
            self.assertEqual(result['pooled_rmse'],second['pooled_rmse'])
            self.assertEqual(files,{p.name:(digest(p),p.stat().st_mtime_ns) for p in (Path(d)/'screen').glob('fold_*.npz')})
            with self.assertRaises(ValueError):screen_arrays(data,Path(d),repeats=30,max_rows=512,signature='changed')

if __name__=='__main__':unittest.main(verbosity=2)
