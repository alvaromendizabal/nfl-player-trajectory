import copy,inspect,json,unittest
import numpy as np
from fixtures import meta_fixture
from history_features import (Metadata,from_arrays,fit_ordered,transform_frozen,new_state,
    validate_state,state_hash,NAMES,CONFIG,group_keys,SUPPORT_NAMES,ordinal_dates,support_summary)

class HistoryTests(unittest.TestCase):
    def setUp(self):
        m,y=meta_fixture();self.m,self.y=m,y
        days=np.unique(m.keys[:,0]//100);self.tr=np.flatnonzero(m.keys[:,0]//100<days[-2]);self.va=np.flatnonzero(m.keys[:,0]//100>=days[-2])
        self.a=m.take(self.tr);self.b=m.take(self.va);self.target=y[self.tr]
    def run_features(self,y=None):return fit_ordered(self.a,self.target if y is None else y,self.b)
    def test_names(self):self.assertEqual(len(NAMES),21);self.assertEqual(len(set(NAMES)),21)
    def test_shape_finite(self):
        x,z,s,r=self.run_features();self.assertEqual(x.shape,(len(self.tr),21));self.assertTrue(np.isfinite(z).all())
    def test_first_day_cold(self):
        x,*_=self.run_features();ii=self.a.keys[:,0]//100==(self.a.keys[:,0]//100).min();self.assertTrue(np.all(x[ii][:,[1,4,7]]==1))
    def test_first_day_mean_zero(self):
        x,*_=self.run_features();ii=self.a.keys[:,0]//100==(self.a.keys[:,0]//100).min();self.assertTrue(np.all(x[ii][:,[9,10,13,14,15,16,19,20]]==0))
    def test_same_date_target_poison(self):
        x,*_=self.run_features();yy=self.target.copy();day=np.unique(self.a.keys[:,0]//100)[1];mask=self.a.keys[:,0]//100==day;yy[mask]+=9999
        z,*_=self.run_features(yy);np.testing.assert_array_equal(x[mask],z[mask])
    def test_same_date_different_games_excluded(self):
        x,*_=self.run_features();day=np.unique(self.a.keys[:,0]//100)[1];ii=np.flatnonzero(self.a.keys[:,0]//100==day)
        # At this date, support is unchanged between both games for the same player/phase.
        ix=[i for i in ii if self.a.keys[i,2]==1 and self.a.keys[i,3]==1];np.testing.assert_array_equal(x[ix[0]],x[ix[1]])
    def test_future_poison(self):
        x,*_=self.run_features();yy=self.target.copy();yy[self.a.keys[:,0]//100==(self.a.keys[:,0]//100).max()]*=777
        z,*_=self.run_features(yy);np.testing.assert_array_equal(x,z)
    def test_earlier_targets_can_change_later(self):
        x,*_=self.run_features();yy=self.target.copy();yy[0]+=100;z,*_=self.run_features(yy);self.assertFalse(np.array_equal(x,z))
    def test_support_independent_of_targets(self):
        x,z,*_=self.run_features();a,b,*_=self.run_features(self.target*1000+44);np.testing.assert_array_equal(x[:,:9],a[:,:9]);np.testing.assert_array_equal(z[:,:9],b[:,:9])
    def test_frozen_evaluation_no_target_argument(self):
        self.assertEqual(list(inspect.signature(transform_frozen).parameters),['meta','state'])
    def test_evaluation_batch_invariance(self):
        _,z,s,_=self.run_features();other=transform_frozen(self.b.take(np.array([4,2,8])),s);np.testing.assert_array_equal(z[[4,2,8]],other)
    def test_state_immutable_in_prediction(self):
        _,_,s,_=self.run_features();h=state_hash(s);transform_frozen(self.b,s);self.assertEqual(state_hash(s),h)
    def test_train_inputs_unchanged(self):
        old=self.a.keys.copy();y=self.target.copy();self.run_features();np.testing.assert_array_equal(old,self.a.keys);np.testing.assert_array_equal(y,self.target)
    def test_train_permutation_invariance(self):
        x,z,s,_=self.run_features();perm=np.random.default_rng(3).permutation(len(self.a.keys))
        a,b,t,_=fit_ordered(self.a.take(perm),self.target[perm],self.b);np.testing.assert_array_equal(a,x[perm]);np.testing.assert_array_equal(b,z);self.assertEqual(s,t)
    def test_eval_permutation_invariance(self):
        _,z,s,_=self.run_features();perm=np.arange(len(self.b.keys))[::-1];np.testing.assert_array_equal(transform_frozen(self.b.take(perm),s),z[perm])
    def test_serialization_replay(self):
        _,z,s,_=self.run_features();np.testing.assert_array_equal(transform_frozen(self.b,json.loads(json.dumps(s))),z)
    def test_heldout_same_day_rejected(self):
        with self.assertRaises(ValueError):fit_ordered(self.a,self.target,self.a.take(np.array([0])))
    def test_frozen_current_date_rejected(self):
        _,_,s,_=self.run_features()
        with self.assertRaises(ValueError):transform_frozen(self.a,s)
    def test_unknown_player_falls_back(self):
        _,_,s,_=self.run_features();k=self.b.keys.copy();k[:,2]+=999;b=Metadata(k,self.b.role,self.b.time,self.b.horizon);z=transform_frozen(b,s)
        self.assertTrue(np.all(z[:,7]==1));np.testing.assert_array_equal(z[:,9:15],z[:,15:])
    def test_player_ids_are_keys_not_covariates(self):
        x,z,*_=self.run_features();ka=self.a.keys.copy();kb=self.b.keys.copy();ka[:,2]+=100;kb[:,2]+=100
        a,b,*_=fit_ordered(Metadata(ka,self.a.role,self.a.time,self.a.horizon),self.target,Metadata(kb,self.b.role,self.b.time,self.b.horizon));np.testing.assert_array_equal(a,x);np.testing.assert_array_equal(b,z)
    def test_reflection_covariance(self):
        x,z,*_=self.run_features();yy=self.target.copy();yy[:,1]*=-1;a,b,*_=self.run_features(yy)
        expected=x.copy();expected[:,[10,14,16,20]]*=-1;np.testing.assert_allclose(a,expected,rtol=0,atol=1e-13)
    def test_donor_unit_not_forecast_rows(self):
        _,_,s,r=self.run_features();# 4 days * 2 games * 3 players * 3 phases
        self.assertEqual(r['trajectory_phase_donors'],4*2*3*3)
        self.assertEqual(sum(v[0] for v in s['tables']['global_phase'].values()),4*2*3*3)
    def test_expected_residual_units(self):
        _,z,*_=self.run_features();np.testing.assert_array_equal(z[:,13:15],z[:,9:11]*self.b.time[:,None])
    def test_duplicate_key_rejected(self):
        k=self.a.keys.copy();k[1]=k[0]
        with self.assertRaises(ValueError):Metadata(k,self.a.role,self.a.time,self.a.horizon).validate()
    def test_bad_date_rejected(self):
        with self.assertRaises(ValueError):ordinal_dates(np.array([2023139900]))
    def test_role_change_rejected(self):
        r=self.a.role.copy();r[1]=3
        with self.assertRaises(ValueError):Metadata(self.a.keys,r,self.a.time,self.a.horizon).validate()
    def test_horizon_change_rejected(self):
        h=self.a.horizon.copy();h[1]+=1
        with self.assertRaises(ValueError):Metadata(self.a.keys,self.a.role,self.a.time,h).validate()
    def test_nan_target_rejected(self):
        yy=self.target.copy();yy[0,0]=np.nan
        with self.assertRaises(ValueError):self.run_features(yy)
    def test_config_drift_rejected(self):
        s=new_state();s['config']['zero_history_rate_sd']=4
        with self.assertRaises(ValueError):validate_state(s)
    def test_corrupt_table_rejected(self):
        _,_,s,_=self.run_features();next(iter(s['tables']['global_phase'].values()))[0]=-1
        with self.assertRaises(ValueError):validate_state(s)
    def test_frozen_return_not_alias(self):
        _,z,s,_=self.run_features();h=state_hash(s);z[:]=1;self.assertEqual(state_hash(s),h)
    def test_empty_evaluation(self):
        _,z,_,r=fit_ordered(self.a,self.target,self.a.take(np.array([],int)));self.assertEqual(z.shape,(0,21));self.assertEqual(r['evaluation_rows'],0)
    def test_metadata_float32_roundtrip(self):
        b=np.ones((len(self.m.keys),72));b[:,1]=self.m.time;b[:,3]=self.m.horizon-self.m.time;b[:,4]=self.m.time/self.m.horizon;b[:,5]=0
        m=from_arrays(self.m.keys,self.m.role,b.astype(np.float32));np.testing.assert_array_equal(m.horizon,self.m.horizon)
    def test_offset_rejected(self):
        b=np.ones((len(self.m.keys),72))
        with self.assertRaises(ValueError):from_arrays(self.m.keys,self.m.role,b)
    def test_query_clock_rejected(self):
        with self.assertRaises(ValueError):Metadata(self.a.keys,self.a.role,self.a.time+1,self.a.horizon+2).validate()
    def test_support_summary_shape(self):
        with self.assertRaises(ValueError):support_summary(np.zeros((4,7)))
    def test_rate_quantiles_finite(self):
        *_,r=self.run_features();self.assertTrue(np.isfinite(r['training_rate_abs_quantiles']).all())
    def test_target_shape_rejected(self):
        with self.assertRaises(ValueError):self.run_features(self.target[:,0])
    def test_phase_end_in_last_bin(self):
        _,b,_=group_keys(self.a);self.assertTrue(b[11].split(':')[1]=='2')
    def test_metadata_nan_rejected(self):
        h=self.a.horizon.copy();h[0]=np.nan
        with self.assertRaises(ValueError):Metadata(self.a.keys,self.a.role,self.a.time,h).validate()
    def test_readonly_arrays_accepted(self):
        self.a.keys.flags.writeable=False;self.target.flags.writeable=False;self.run_features()
    def test_game_dates_age_real_days(self):
        _,z,*_=self.run_features();age=np.expm1(z[:,2]);self.assertTrue(set(np.rint(age))=={7.,14.})

if __name__=='__main__':unittest.main()
