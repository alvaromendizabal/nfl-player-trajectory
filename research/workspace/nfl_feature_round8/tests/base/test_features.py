import inspect,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]));sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fixtures import raw_play
from sequence_features import build,terminal_view,NODE_NAMES,PAIR_NAMES
from sequence_data import storage_parity

class Features(unittest.TestCase):
    def setUp(self):self.raw=raw_play();self.f=build(self.raw,origin=20)
    def test_shapes(self):
        self.assertEqual(self.f['node'].shape,(4,20,10));self.assertEqual(self.f['pair'].shape,(4,4,20,8))
    def test_names_unique(self):self.assertEqual(len(set(NODE_NAMES)),10);self.assertEqual(len(set(PAIR_NAMES)),8)
    def test_finite(self):self.assertTrue(all(np.isfinite(v).all() for v in self.f.values()))
    def test_immutable(self):
        old=self.raw.copy(deep=True);build(self.raw,origin=20);__import__('pandas').testing.assert_frame_equal(old,self.raw)
    def test_reproducible(self):
        other=build(self.raw,origin=20)
        for k in self.f:self.assertTrue(np.array_equal(self.f[k],other[k]))
    def test_input_row_order(self):
        other=build(self.raw.sample(frac=1,random_state=2),origin=20)
        for k in self.f:self.assertTrue(np.array_equal(self.f[k],other[k]))
    def test_forecast_labels_not_arguments(self):self.assertEqual(set(inspect.signature(build).parameters),{'observed','origin'})
    def test_unused_target_columns_ignored(self):
        r=self.raw.assign(target_x=np.arange(len(self.raw))*100,future_y=-999)
        for k,v in build(r,origin=20).items():self.assertTrue(np.array_equal(v,self.f[k]))
    def test_future_rows_rejected(self):
        with self.assertRaises(ValueError):build(self.raw,origin=19)
    def test_duplicate_keys(self):
        with self.assertRaises(ValueError):build(__import__('pandas').concat([self.raw,self.raw.iloc[:1]]),origin=20)
    def test_unknown_role(self):
        r=self.raw.copy();r['player_role']='bad'
        with self.assertRaises(ValueError):build(r,origin=20)
    def test_missing_required(self):
        with self.assertRaises(ValueError):build(self.raw.drop(columns='x'),origin=20)
    def test_nan_position(self):
        r=self.raw.copy();r.loc[0,'x']=np.nan
        with self.assertRaises(ValueError):build(r,origin=20)
    def test_many_players(self):
        with self.assertRaises(ValueError):build(raw_play(n=23),origin=20)
    def test_no_recent_player(self):
        with self.assertRaises(ValueError):build(self.raw,origin=99)
    def test_self_edges(self):
        self.assertFalse(self.f['pair_valid'][np.arange(4),np.arange(4)].any())
    def test_actual_gaps(self):
        r=self.raw.loc[~((self.raw.nfl_id==100)&(self.raw.frame_id==8))];f=build(r,origin=20)
        self.assertFalse(f['pair_valid'][0,:,7].any());self.assertFalse(f['node_valid'][0,7].any())
    def test_no_velocity_gap_bridge(self):
        r=self.raw.drop(columns=['s','dir']);r=r.loc[~((r.nfl_id==100)&(r.frame_id==8))];f=build(r,origin=20)
        self.assertFalse(f['node_valid'][0,8,2]);self.assertTrue(f['node_valid'][0,9,2])
    def test_missing_orientation(self):
        f=build(self.raw.drop(columns='o'),origin=20);self.assertFalse(f['node_valid'][...,6:8].any())
    def test_stationary_alignment(self):
        r=self.raw.copy();r['s']=0;f=build(r,origin=20);self.assertFalse(f['pair_valid'][...,7].any())
    def test_relative_units(self):
        i=self.raw.loc[(self.raw.nfl_id==100)&(self.raw.frame_id==20)].iloc[0];j=self.raw.loc[(self.raw.nfl_id==101)&(self.raw.frame_id==20)].iloc[0]
        self.assertAlmostEqual(float(self.f['pair'][0,1,-1,0]),(j.x-i.x)/20,places=6)
    def test_translation(self):
        r=self.raw.copy()
        for c in ['x','ball_land_x']:r[c]+=7
        for c in ['y','ball_land_y']:r[c]-=3
        other=build(r,origin=20)
        np.testing.assert_allclose(other['pair'],self.f['pair'],atol=1e-6)
    def test_direction_reversal(self):
        r=self.raw.copy();r['play_direction']='left'
        for c in ['x','ball_land_x']:r[c]=120-r[c]
        for c in ['y','ball_land_y']:r[c]=160/3-r[c]
        for c in ['dir','o']:r[c]=(r[c]+180)%360
        other=build(r,origin=20);np.testing.assert_allclose(other['node'],self.f['node'],atol=1e-6);np.testing.assert_allclose(other['pair'],self.f['pair'],atol=1e-6)
    def test_terminal_masks_unchanged(self):
        p=terminal_view(self.f['pair'],self.f['pair_valid']);self.assertTrue(np.all(p[~self.f['pair_valid']]==0))
    def test_terminal_last_value(self):
        p=terminal_view(self.f['pair'],self.f['pair_valid']);np.testing.assert_array_equal(p[:,:,-1],self.f['pair'][:,:,-1])
    def test_history_differs(self):self.assertFalse(np.array_equal(terminal_view(self.f['pair'],self.f['pair_valid']),self.f['pair']))
    def test_terminal_poison_masked(self):
        v=self.f['pair'].copy();v[~self.f['pair_valid']]=np.nan
        np.testing.assert_array_equal(terminal_view(v,self.f['pair_valid']),terminal_view(self.f['pair'],self.f['pair_valid']))
    def test_terminal_invalid_valid_nan(self):
        v=self.f['pair'].copy();v[self.f['pair_valid']]=np.nan
        with self.assertRaises(ValueError):terminal_view(v,self.f['pair_valid'])
    def test_terminal_mask_type(self):
        with self.assertRaises(ValueError):terminal_view(self.f['pair'],self.f['pair_valid'].astype(int))
    def test_terminal_shape(self):
        with self.assertRaises(ValueError):terminal_view(self.f['pair'][:1],self.f['pair_valid'])
    def test_storage_float32(self):
        r=np.array([[.1,.3]],np.float64);storage_parity(r,r.astype(np.float32))
    def test_storage_float64(self):
        r=np.array([[.1,.3]],np.float64);storage_parity(r,r.copy())
    def test_storage_real_mismatch(self):
        r=np.array([[.1,.3]],np.float64);s=r.astype(np.float32);s[0,0]=np.nextafter(s[0,0],np.float32(1))
        with self.assertRaises(ValueError):storage_parity(r,s)
    def test_storage_shape(self):
        with self.assertRaises(ValueError):storage_parity(np.ones((2,3)),np.ones((3,2)))
    def test_storage_integer_rejected(self):
        with self.assertRaises(ValueError):storage_parity(np.ones((1,2)),np.ones((1,2),dtype=int))
    def test_storage_nan_rejected(self):
        with self.assertRaises(ValueError):storage_parity(np.array([[np.nan]]),np.array([[np.nan]]))
if __name__=='__main__':unittest.main()
