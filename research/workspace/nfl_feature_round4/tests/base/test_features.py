import inspect
import unittest
import numpy as np
import pandas as pd
from fixtures import fixture
from origin_features import ALL_NAMES, WIDTH, STATE_NAMES
from state_features import (build_features,storage_parity,DIRECT_NAMES,GOAL_NAMES,GOAL_ODD)

class FeatureTests(unittest.TestCase):
    def setUp(self):self.raw,self.q=fixture()
    def build(self,raw=None,q=None):
        return build_features(self.raw if raw is None else raw,self.q if q is None else q,cutoff=20)
    def test_shape(self):
        z=self.build();self.assertEqual(z['state'].shape,(30,62));self.assertEqual(z['goal'].shape,(30,26))
    def test_names_unique(self):self.assertEqual(len(set(DIRECT_NAMES+GOAL_NAMES)),88)
    def test_finite(self):
        for key in ('state','goal','legacy'):self.assertTrue(np.isfinite(self.build()[key]).all())
    def test_repeat_exact(self):
        a,b=self.build(),self.build()
        for k in a:np.testing.assert_array_equal(a[k],b[k])
    def test_immutable(self):
        before=self.raw.copy(deep=True);q=self.q.copy();self.build()
        pd.testing.assert_frame_equal(self.raw,before);np.testing.assert_array_equal(self.q,q)
    def test_row_permutation(self):
        b=self.build(self.raw.sample(frac=1,random_state=7))
        for k,v in self.build().items():np.testing.assert_array_equal(v,b[k])
    def test_query_permutation(self):
        b=self.build(q=self.q[::-1])
        for k,v in self.build().items():np.testing.assert_array_equal(v[::-1],b[k])
    def test_observed_state_independent_of_query_time(self):
        z=self.build();np.testing.assert_array_equal(z['state'][:15],np.repeat(z['state'][:1],15,axis=0))
        self.assertNotEqual(z['legacy'][0,6],z['legacy'][10,6])
    def test_role_is_direct_indicator(self):
        z=self.build();ix=DIRECT_NAMES.index('observed__role_Targeted Receiver')
        np.testing.assert_array_equal(z['state'][:15,ix],np.ones(15))
        np.testing.assert_array_equal(z['state'][15:,ix],np.zeros(15))
    def test_receiver_distance_units(self):
        z=self.build();ix=GOAL_NAMES.index('receiver_distance')
        self.assertAlmostEqual(z['goal'][15,ix],np.sqrt(8)/20)
    def test_goal_distance_units(self):
        self.assertAlmostEqual(self.build()['goal'][0,GOAL_NAMES.index('goal_distance')],9/20)
    def test_radial_velocity_units(self):
        self.assertAlmostEqual(self.build()['goal'][0,GOAL_NAMES.index('speed_toward_goal')],.3)
    def test_required_speed_units(self):
        self.assertAlmostEqual(self.build()['goal'][0,GOAL_NAMES.index('required_average_speed')],.6)
    def test_forecast_projection(self):
        z=self.build();self.assertAlmostEqual(z['goal'][9,GOAL_NAMES.index('cv_distance_at_query')],6/20)
    def test_endpoint_projection(self):
        self.assertAlmostEqual(self.build()['goal'][0,GOAL_NAMES.index('cv_distance_at_endpoint')],4.5/20)
    def test_closest_time_capped_to_supplied_horizon(self):
        self.assertAlmostEqual(self.build()['goal'][0,GOAL_NAMES.index('closest_approach_time')],1.5/5)
    def test_stationary_support(self):
        r=self.raw.copy();r.s=0;z=self.build(r)
        self.assertEqual(z['goal'][:,GOAL_NAMES.index('heading_present')].sum(),0)
        self.assertEqual(z['goal'][:,GOAL_NAMES.index('closest_approach_present')].sum(),0)
    def test_coincident_goal(self):
        r=self.raw.copy();r.ball_land_x=36.;r.ball_land_y=20.;z=self.build(r)
        self.assertEqual(z['goal'][0,GOAL_NAMES.index('goal_axis_present')],0)
        self.assertTrue(np.isfinite(z['goal']).all())
    def test_missing_velocity(self):
        r=self.raw.drop(columns=['s','dir']).loc[self.raw.frame_id==20].copy();z=self.build(r)
        self.assertEqual(z['goal'][:,GOAL_NAMES.index('velocity_present')].sum(),0)
    def test_adjacent_velocity_fallback(self):
        z=self.build(self.raw.drop(columns=['s','dir']))
        self.assertEqual(z['goal'][:,GOAL_NAMES.index('velocity_present')].sum(),len(self.q))
    def test_no_velocity_gap_bridge(self):
        r=self.raw.drop(columns=['s','dir']).loc[self.raw.frame_id.isin([18,20])].copy();z=self.build(r)
        self.assertEqual(z['goal'][:,GOAL_NAMES.index('velocity_present')].sum(),0)
    def test_future_rejected(self):
        r=self.raw.copy();r.loc[r.index[-1],'frame_id']=21
        with self.assertRaises(ValueError):self.build(r)
    def test_unrelated_future_columns_ignored(self):
        r=self.raw.copy();r['future_x']=1e20;r['truth']=np.nan
        for k,v in self.build().items():np.testing.assert_array_equal(v,self.build(r)[k])
    def test_no_target_interface(self):
        self.assertNotIn('truth',inspect.signature(build_features).parameters)
        self.assertNotIn('y',inspect.signature(build_features).parameters)
    def test_duplicate_raw(self):
        with self.assertRaises(ValueError):self.build(pd.concat([self.raw,self.raw.iloc[:1]]))
    def test_duplicate_query(self):
        with self.assertRaises(ValueError):self.build(q=np.r_[self.q,self.q[:1]])
    def test_wrong_play(self):
        q=self.q.copy();q[:,1]+=1
        with self.assertRaises(ValueError):self.build(q=q)
    def test_bad_horizon(self):
        q=self.q.copy();q[-1,3]=100
        with self.assertRaises(ValueError):self.build(q=q)
    def test_float_keys_rejected(self):
        with self.assertRaises(ValueError):self.build(q=self.q.astype(float))
    def test_unscored_query_rejected(self):
        q=self.q.copy();q[:15,2]=3
        with self.assertRaises(ValueError):self.build(q=q)
    def test_id_values_not_predictors(self):
        r=self.raw.copy();r.nfl_id+=500;q=self.q.copy();q[:,2]+=500
        a,b=self.build(),self.build(r,q)
        for k in ('state','goal','legacy'):np.testing.assert_array_equal(a[k],b[k])
    def test_translation_goal_invariance(self):
        r=self.raw.copy();r.x+=7;r.y+=3;r.ball_land_x+=7;r.ball_land_y+=3
        np.testing.assert_allclose(self.build()['goal'],self.build(r)['goal'],rtol=0,atol=1e-12)
    def test_lateral_reflection(self):
        r=self.raw.copy();r.y=WIDTH-r.y;r.ball_land_y=WIDTH-r.ball_land_y
        r['dir']=(180-r['dir'])%360;r.o=(180-r.o)%360
        want=self.build()['goal'].copy();want[:,GOAL_ODD]*=-1
        np.testing.assert_allclose(want,self.build(r)['goal'],rtol=0,atol=1e-12)
    def test_direction_normalization(self):
        r=self.raw.copy();r.x=120-r.x;r.y=WIDTH-r.y;r.ball_land_x=120-r.ball_land_x;r.ball_land_y=WIDTH-r.ball_land_y
        r['dir']=(r['dir']+180)%360;r.o=(r.o+180)%360;r.play_direction='left'
        for k in ('goal','state'):np.testing.assert_allclose(self.build()[k],self.build(r)[k],rtol=0,atol=1e-12)
    def test_nonfinite_coordinates(self):
        r=self.raw.copy();r.loc[0,'x']=np.nan
        with self.assertRaises(ValueError):self.build(r)
    def test_float32_storage_exact(self):
        a=self.build()['legacy'];storage_parity(a,a.astype(np.float32))
    def test_float64_storage_exact(self):
        a=self.build()['legacy'];storage_parity(a,a.copy())
    def test_float32_one_ulp_change_rejected(self):
        a=self.build()['legacy'];b=a.astype(np.float32);b[0,1]=np.nextafter(b[0,1],np.float32(np.inf))
        with self.assertRaises(ValueError):storage_parity(a,b)
    def test_float64_one_ulp_change_rejected(self):
        a=self.build()['legacy'];b=a.copy();b[0,1]=np.nextafter(b[0,1],np.inf)
        with self.assertRaises(ValueError):storage_parity(a,b)
    def test_old_float64_vs_float32_guard_would_fail(self):
        a=self.build()['legacy'];b=a.astype(np.float32)
        self.assertFalse(np.allclose(a,b,rtol=0,atol=1e-12));storage_parity(a,b)
    def test_parity_bad_shape(self):
        a=self.build()['legacy']
        with self.assertRaises(ValueError):storage_parity(a,a[:,:-1])
    def test_parity_bad_dtype(self):
        a=self.build()['legacy']
        with self.assertRaises(ValueError):storage_parity(a,a.astype(np.float16))
    def test_parity_nonfinite(self):
        a=self.build()['legacy'];b=a.copy();b[0,0]=np.nan
        with self.assertRaises(ValueError):storage_parity(a,b)

if __name__=='__main__':unittest.main()
