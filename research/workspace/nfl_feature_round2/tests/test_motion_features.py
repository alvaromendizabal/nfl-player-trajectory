from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_features import NAMES, FEATURES, FAMILIES, turn_correction, speed_correction, observed_rates, build_features
from origin_features import KEYS, WIDTH
from fixtures import play


class PhysicsTests(unittest.TestCase):
    def test_zero_turn_exact_cv(self):
        np.testing.assert_array_equal(turn_correction([4., 2.], 0., 2.), [0., 0.])
    def test_quarter_circle_independent_formula(self):
        # v=(2,0), omega=1 rad/s, t=pi/2: absolute displacement=(2,2).
        np.testing.assert_allclose(turn_correction([2.,0.],1.,np.pi/2), [2.-np.pi, 2.], atol=1e-12)
    def test_small_turn_limit(self):
        np.testing.assert_allclose(turn_correction([2.,0.],1e-12,1.), [0.,1e-12], atol=1e-20)
    def test_negative_turn(self):
        a=turn_correction([2.,0.],1.,1.); b=turn_correction([2.,0.],-1.,1.)
        np.testing.assert_allclose(b, a*[1,-1], atol=1e-12)
    def test_turn_reflection(self):
        np.testing.assert_allclose(turn_correction([3.,-2.],-.5,2.),turn_correction([3.,2.],.5,2.)*[1,-1])
    def test_stationary_turn(self):
        np.testing.assert_array_equal(turn_correction([0.,0.],2.,3.),[0.,0.])
    def test_speed_zero_acceleration(self):
        np.testing.assert_array_equal(speed_correction([4.,0.],0.,2.),[0.,0.])
    def test_speed_increase(self):
        np.testing.assert_allclose(speed_correction([4.,0.],2.,2.),[4.,0.])
    def test_braking_stop_no_reverse(self):
        # v=4, a=-2, stop at t=2; traveled 4 yards vs CV=12 at t=3.
        np.testing.assert_allclose(speed_correction([4.,0.],-2.,3.),[-8.,0.])
    def test_braking_before_stop(self):
        np.testing.assert_allclose(speed_correction([4.,0.],-2.,1.),[-1.,0.])
    def test_braking_at_stop_continuity(self):
        a=speed_correction([4.,0.],-2.,2.);b=speed_correction([4.,0.],-2.,2.+1e-8)
        np.testing.assert_allclose(a,b,atol=5e-8)
    def test_zero_speed_has_no_direction(self):
        np.testing.assert_array_equal(speed_correction([0.,0.],1.,5.),[0.,0.])
    def test_invalid_physics(self):
        for fn in (turn_correction,speed_correction):
            for v,a,t in [([np.nan,0],0,1),([1,0],np.inf,1),([1,0],0,-1),([1],0,1)]:
                with self.subTest(fn=fn,v=v,a=a,t=t):
                    with self.assertRaises(ValueError):fn(v,a,t)
    def test_rate_known_radians_per_second(self):
        f=np.arange(1,6);angle=np.arange(5)*.1
        v=2*np.column_stack([np.cos(angle),np.sin(angle)])
        r=observed_rates(f,v,np.ones(5,bool),5,5)
        self.assertAlmostEqual(r['omega'],1.,places=12);self.assertAlmostEqual(r['turn_support'],1.)
    def test_rate_known_acceleration_units(self):
        v=np.column_stack([2+np.arange(5)*.2,np.zeros(5)])
        r=observed_rates(np.arange(1,6),v,np.ones(5,bool),5,5)
        self.assertAlmostEqual(r['acceleration'],2.,places=12)
    def test_gap_not_bridged(self):
        r=observed_rates(np.array([1,2,5]),np.array([[1.,0],[3.,0],[100.,0]]),np.ones(3,bool),5,5)
        self.assertEqual(r['turn_support'],0.);self.assertEqual(r['brake_support'],0.)
    def test_missing_velocity_poison_ignored(self):
        r=observed_rates(np.arange(1,6),np.array([[1.,0],[np.nan,np.inf],[1.,0],[1.,0],[1.,0]]),np.array([True,False,True,True,True]),5,5)
        self.assertEqual(r['omega'],0.);self.assertEqual(r['turn_support'],.5)
    def test_no_terminal_velocity_blocks_rate(self):
        r=observed_rates(np.arange(1,6),np.zeros((5,2)),np.array([True,True,True,True,False]),5,5)
        self.assertEqual(r['brake_support'],0)
    def test_rate_invalid_mask_rejected(self):
        with self.assertRaises(ValueError):observed_rates(np.arange(1,6),np.zeros((5,2)),np.ones(5),5,5)
    def test_rate_future_rejected(self):
        with self.assertRaises(ValueError):observed_rates(np.arange(1,6),np.zeros((5,2)),np.ones(5,bool),4,5)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.raw,self.labels=play(); self.requests=self.labels[KEYS].copy()
    def test_sixty_unique_features(self):
        self.assertEqual(len(NAMES),60);self.assertEqual(len(set(NAMES)),60)
        self.assertEqual({k:len(v) for k,v in FAMILIES.items()},{'turn':24,'brake':24,'orientation':12})
    def test_dimensions_and_finite(self):
        x,support=build_features(self.raw,self.requests)
        self.assertEqual(x.shape,(32,60));self.assertTrue(np.isfinite(x).all());self.assertEqual(len(support),10)
    def test_requests_cannot_include_targets(self):
        with self.assertRaises(ValueError):build_features(self.raw,self.labels)
    def test_no_target_argument(self):
        self.assertEqual(set(inspect.signature(build_features).parameters),{'observed','requests','origin'})
    def test_future_observed_rows_blocked(self):
        with self.assertRaises(ValueError):build_features(self.raw,self.requests,origin=29)
    def test_unrelated_columns_ignored(self):
        a,_=build_features(self.raw,self.requests)
        raw=self.raw.assign(truth_x=np.inf,truth_y=np.nan,parent_prediction=-1e9)
        b,_=build_features(raw,self.requests);np.testing.assert_array_equal(a,b)
    def test_input_immutability(self):
        raw=self.raw.copy(deep=True);q=self.requests.copy(deep=True)
        build_features(self.raw,self.requests)
        pd.testing.assert_frame_equal(raw,self.raw);pd.testing.assert_frame_equal(q,self.requests)
    def test_raw_order_invariance(self):
        a,_=build_features(self.raw,self.requests);b,_=build_features(self.raw.sample(frac=1,random_state=9),self.requests)
        np.testing.assert_array_equal(a,b)
    def test_query_order_equivariance(self):
        a,_=build_features(self.raw,self.requests);order=np.arange(len(a))[::-1]
        b,_=build_features(self.raw,self.requests.iloc[order]);np.testing.assert_array_equal(a[order],b)
    def test_identifier_not_numeric_covariate(self):
        a,_=build_features(self.raw,self.requests);raw=self.raw.copy();q=self.requests.copy()
        raw.nfl_id+=1000;q.nfl_id+=1000
        b,_=build_features(raw,q);np.testing.assert_array_equal(a,b)
    def test_translation_invariance(self):
        a,_=build_features(self.raw,self.requests);raw=self.raw.copy()
        raw.x+=2;raw.y+=3;raw.ball_land_x+=2;raw.ball_land_y+=3
        b,_=build_features(raw,self.requests);np.testing.assert_allclose(a,b,atol=1e-12)
    def test_lateral_reflection(self):
        a,_=build_features(self.raw,self.requests);raw=self.raw.copy()
        raw.y=WIDTH-raw.y;raw.ball_land_y=WIDTH-raw.ball_land_y
        raw.dir=(180-raw.dir)%360;raw.o=(180-raw.o)%360
        b,_=build_features(raw,self.requests)
        sign=np.array([-1 if f.component=='dy' else 1 for f in FEATURES])
        np.testing.assert_allclose(a*sign,b,atol=1e-12)
    def test_play_direction_canonical_parity(self):
        a,_=build_features(self.raw,self.requests);raw=self.raw.copy()
        raw.x=120-raw.x;raw.y=WIDTH-raw.y;raw.ball_land_x=120-raw.ball_land_x;raw.ball_land_y=WIDTH-raw.ball_land_y
        raw.dir=(raw.dir+180)%360;raw.o=(raw.o+180)%360;raw.play_direction='left'
        b,_=build_features(raw,self.requests);np.testing.assert_allclose(a,b,atol=1e-12)
    def test_absent_orientation_is_not_zero_measurement(self):
        x,_=build_features(self.raw.drop(columns='o'),self.requests)
        np.testing.assert_array_equal(x[:,FAMILIES['orientation']],0)
    def test_nonfinite_optional_orientation_masks(self):
        raw=self.raw.copy();raw.o=np.nan;x,_=build_features(raw,self.requests)
        self.assertTrue(np.isfinite(x).all());np.testing.assert_array_equal(x[:,FAMILIES['orientation']],0)
    def test_position_velocity_fallback(self):
        x,_=build_features(self.raw.drop(columns=['s','dir']),self.requests);self.assertTrue(np.isfinite(x).all())
    def test_unknown_role_rejected(self):
        raw=self.raw.copy();raw['player_role']='unknown'
        with self.assertRaises(ValueError):build_features(raw,self.requests)
    def test_future_horizon_violation(self):
        q=self.requests.copy();q.iloc[0,q.columns.get_loc('frame_id')]=100
        with self.assertRaises(ValueError):build_features(self.raw,q)
    def test_duplicate_query_rejected(self):
        with self.assertRaises(ValueError):build_features(self.raw,pd.concat([self.requests,self.requests.iloc[:1]]))
    def test_unknown_query_player_rejected(self):
        q=self.requests.copy();q.nfl_id=999
        with self.assertRaises(ValueError):build_features(self.raw,q)
    def test_fractional_query_time_rejected(self):
        q=self.requests.astype(float);q.loc[0,'frame_id']=1.5
        with self.assertRaises(ValueError):build_features(self.raw,q)
    def test_nonfinite_raw_position_rejected(self):
        raw=self.raw.copy();raw.loc[0,'x']=np.nan
        with self.assertRaises(ValueError):build_features(raw,self.requests)
    def test_one_frame_history_graceful(self):
        raw=self.raw[self.raw.frame_id==30];x,_=build_features(raw,self.requests)
        np.testing.assert_array_equal(x[:,np.r_[FAMILIES['turn'],FAMILIES['brake']]],0)
    def test_all_stationary_graceful(self):
        raw=self.raw.copy();raw.s=0
        x,_=build_features(raw,self.requests);np.testing.assert_array_equal(x,0)
    def test_stale_age_included(self):
        raw=self.raw[~((self.raw.nfl_id==100)&(self.raw.frame_id==30))]
        x,s=build_features(raw,self.requests);self.assertTrue(np.isfinite(x).all())
        self.assertTrue(any(r['age_seconds']==.1 for r in s))
    def test_unscored_role_blocks_query(self):
        q=self.requests.iloc[:1].copy();q.nfl_id=102
        with self.assertRaises(ValueError):build_features(self.raw,q)
    def test_forecast_mask_is_role_specific(self):
        x,_=build_features(self.raw,self.requests)
        for i,ident in enumerate(self.requests.nfl_id):
            role='Targeted Receiver' if ident==100 else 'Defensive Coverage'
            wrong=[j for j,f in enumerate(FEATURES) if f.role!=role]
            np.testing.assert_array_equal(x[i,wrong],0)
