import unittest
import numpy as np
from fixtures import fixture
from pass_features import build_features,STATIC_NAMES,HISTORY_NAMES,NAMES,ODD_STATIC,ODD_HISTORY
from origin_features import WIDTH

class FeatureTests(unittest.TestCase):
    def setUp(self):self.raw,self.keys=fixture()
    def build(self,r=None,k=None):return build_features(self.raw if r is None else r,self.keys if k is None else k,cutoff=20)
    def test_schema(self):
        z=self.build();self.assertEqual(z['static'].shape,(30,21));self.assertEqual(z['history'].shape,(30,27));self.assertEqual(len(set(NAMES)),48)
    def test_finite(self):
        for k in ('static','history'):self.assertTrue(np.isfinite(self.build()[k]).all())
    def test_physics_axis(self):
        z=self.build()['static'][15];self.assertAlmostEqual(z[0],2);self.assertAlmostEqual(z[1],.6);self.assertAlmostEqual(z[2],.2)
    def test_physics_velocity(self):
        z=self.build()['static'][15];self.assertAlmostEqual(z[5],.2);self.assertAlmostEqual(z[6],0)
    def test_exact_offset_displacement(self):self.assertAlmostEqual(self.build()['history'][15,0],.1)
    def test_constant_velocity_change(self):self.assertAlmostEqual(self.build()['history'][15,2],0)
    def test_query_independent_static(self):self.assertTrue(np.array_equal(self.build()['static'][0],self.build()['static'][14]))
    def test_row_order(self):
        a=self.build();b=self.build(self.raw.sample(frac=1,random_state=8))
        for k in a:np.testing.assert_array_equal(a[k],b[k])
    def test_query_order(self):
        a=self.build();b=self.build(k=self.keys[::-1])
        for k in a:np.testing.assert_array_equal(a[k][::-1],b[k])
    def test_translation(self):
        r=self.raw.copy();r[['x','ball_land_x']]+=7;r[['y','ball_land_y']]+=4
        for k in ('static','history'):np.testing.assert_allclose(self.build()[k],self.build(r)[k],atol=1e-14)
    def test_rotation(self):
        r=self.raw.copy();theta=.72;c,s=np.cos(theta),np.sin(theta);R=np.array([[c,-s],[s,c]])
        for cols in (['x','y'],['ball_land_x','ball_land_y']):r[cols]=r[cols].to_numpy()@R.T
        r.dir-=np.rad2deg(theta);r.o-=np.rad2deg(theta)
        for k in ('static','history'):np.testing.assert_allclose(self.build()[k],self.build(r)[k],atol=1e-13)
    def test_lateral_reflection(self):
        r=self.raw.copy();r.y=WIDTH-r.y;r.ball_land_y=WIDTH-r.ball_land_y;r.dir=180-r.dir;r.o=180-r.o
        a=self.build();a['static'][:,ODD_STATIC]*=-1;a['history'][:,ODD_HISTORY]*=-1
        b=self.build(r)
        for k in ('static','history'):np.testing.assert_allclose(a[k],b[k],atol=1e-13)
    def test_left_canonical(self):
        r=self.raw.copy();r.x=120-r.x;r.y=WIDTH-r.y;r.ball_land_x=120-r.ball_land_x;r.ball_land_y=WIDTH-r.ball_land_y
        r.dir=(r.dir+180)%360;r.o=(r.o+180)%360;r.play_direction='left'
        for k in ('static','history'):np.testing.assert_allclose(self.build()[k],self.build(r)[k],atol=1e-13)
    def test_missing_passer(self):
        z=self.build(self.raw[self.raw.nfl_id!=3]);self.assertFalse(z['static'].any());self.assertFalse(z['history'].any())
    def test_stale_passer_not_filled(self):
        r=self.raw[~((self.raw.nfl_id==3)&(self.raw.frame_id==20))];self.assertFalse(self.build(r)['static'].any())
    def test_degenerate_axis(self):
        r=self.raw.copy();r.ball_land_x=10;r.ball_land_y=20;self.assertFalse(self.build(r)['static'].any())
    def test_missing_receiver_support(self):
        r=self.raw.copy();r.loc[r.nfl_id==1,'player_role']='Other Route Runner';z=self.build(r)['static'];self.assertFalse(z[:,19:].any())
    def test_missing_orientation(self):
        z=self.build(self.raw.drop(columns='o'));self.assertFalse(z['static'][:,18].any())
    def test_gap_endpoint_not_filled(self):
        r=self.raw[~((self.raw.nfl_id==2)&(self.raw.frame_id==15))];self.assertFalse(self.build(r)['history'][15,:9].any())
    def test_receiver_gap_support(self):
        r=self.raw[~((self.raw.nfl_id==1)&(self.raw.frame_id==15))];z=self.build(r)['history'][15];self.assertEqual(z[8],0);self.assertEqual(z[6],1)
    def test_no_speed_fallback(self):
        z=self.build(self.raw.drop(columns=['s','dir']));self.assertTrue(z['static'][:,17].all())
    def test_hidden_target_columns_ignored(self):
        r=self.raw.copy();r['future_x']=np.inf;r['future_y']=np.nan;r['truth']=9999
        for k in ('static','history'):np.testing.assert_array_equal(self.build()[k],self.build(r)[k])
    def test_duplicate_query(self):
        with self.assertRaises(ValueError):self.build(k=np.vstack([self.keys,self.keys[0]]))
    def test_duplicate_raw(self):
        import pandas as pd
        with self.assertRaises(ValueError):self.build(pd.concat([self.raw,self.raw.iloc[:1]]))
    def test_future_row(self):
        r=self.raw.copy();r.loc[0,'frame_id']=21
        with self.assertRaises(ValueError):self.build(r)
    def test_wrong_play(self):
        k=self.keys.copy();k[:,1]+=1
        with self.assertRaises(ValueError):self.build(k=k)
    def test_horizon_reject(self):
        k=self.keys.copy();k[0,3]=16
        with self.assertRaises(ValueError):self.build(k=k)
    def test_float_keys_reject(self):
        with self.assertRaises(ValueError):self.build(k=self.keys.astype(float))
    def test_unscored_reject(self):
        k=self.keys.copy();k[:,2]=4
        with self.assertRaises(ValueError):self.build(k=k[:15])
    def test_input_immutable(self):
        import pandas as pd
        r=self.raw.copy(deep=True);k=self.keys.copy();self.build();pd.testing.assert_frame_equal(self.raw,r);np.testing.assert_array_equal(self.keys,k)
    def test_ambiguous_passer(self):
        r=self.raw.copy();r.loc[r.nfl_id==4,'player_role']='Passer'
        with self.assertRaises(ValueError):self.build(r)
    def test_bad_observed_coordinate(self):
        r=self.raw.copy();r.loc[0,'x']=np.nan
        with self.assertRaises(ValueError):self.build(r)
    def test_role_not_an_assignment(self):
        r=self.raw.copy();r.loc[r.nfl_id==2,'nfl_id']=200;k=self.keys.copy();k[k[:,2]==2,2]=200
        for name in ('static','history'):np.testing.assert_array_equal(self.build()[name],self.build(r,k)[name])
