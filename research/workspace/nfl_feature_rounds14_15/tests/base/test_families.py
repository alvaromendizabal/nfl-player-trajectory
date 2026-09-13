"""Analytic tests for both families. Prepared, not executed by the assistant."""
import copy,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fixtures import sample
from families import build,view,NAMES,SCALES,ODD,summarize,difference

class Shared(unittest.TestCase):
    round_no=14
    def test_shape_and_dtype(self):
        f=build(sample(),self.round_no)
        self.assertEqual(f['values'].shape,(4,20,12));self.assertEqual(f['values'].dtype,np.float32)
        self.assertEqual(f['valid'].dtype,np.bool_)
    def test_names_unique(self):self.assertEqual(len(set(NAMES[self.round_no])),12)
    def test_finite(self):self.assertTrue(np.isfinite(build(sample(),self.round_no)['values']).all())
    def test_repeat(self):
        a=build(sample(),self.round_no);b=build(sample(),self.round_no)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
    def test_immutable(self):
        p=sample();old=copy.deepcopy(p)
        for x in p.values():x.flags.writeable=False
        build(p,self.round_no)
        for k in p:np.testing.assert_array_equal(old[k],p[k])
    def test_labels_ignored(self):
        class Guard(dict):
            def __getitem__(self,k):
                if k in ('y','keys','base','query','train','goal'):raise AssertionError(k)
                return super().__getitem__(k)
        build(Guard(sample()),self.round_no)
    def test_bad_role_rejected(self):
        p=sample();p['role'][1]=[1,0,0,0]
        with self.assertRaises(ValueError):build(p,self.round_no)
    def test_masked_poison(self):
        p=sample();p['node_valid'][1,7]=False;p['node'][1,7]=np.nan
        p['pair'][0,0]=np.nan
        self.assertTrue(np.isfinite(build(p,self.round_no)['values']).all())
    def test_valid_poison_rejected(self):
        p=sample();p['node'][1,8,2]=np.nan
        with self.assertRaises(ValueError):build(p,self.round_no)
    def test_masks_same_in_arms(self):
        f=build(sample(),self.round_no)
        a=[view(f,arm) for arm in ('mask','core','full')]
        self.assertFalse(a[0][0].any());self.assertFalse(a[1][0][...,6:].any())
        np.testing.assert_array_equal(a[1][0][...,:6],a[2][0][...,:6])
        for _,m in a:np.testing.assert_array_equal(m,f['valid'])
    def test_permutation(self):
        p=sample();ix=np.array([2,0,3,1]);q=copy.deepcopy(p)
        for k in ('node','node_valid','role','side','ids','node_age'):q[k]=q[k][ix]
        for k in ('pair','pair_valid','pair_age'):q[k]=q[k][ix][:,ix]
        a=build(p,self.round_no);b=build(q,self.round_no)
        np.testing.assert_array_equal(b['values'],a['values'][ix])
        np.testing.assert_array_equal(b['valid'],a['valid'][ix])
    def test_reflection(self):
        p=sample();q=copy.deepcopy(p)
        q['node'][...,[1,3,5,7]]*=-1;q['pair'][...,[1,3,6]]*=-1
        a=build(p,self.round_no);b=build(q,self.round_no)
        want=a['values'].copy();want[...,list(ODD[self.round_no])]*=-1
        np.testing.assert_allclose(want,b['values'],rtol=0,atol=1e-6)
        np.testing.assert_array_equal(a['valid'],b['valid'])
    def test_past_unchanged_by_future_observations(self):
        p=sample();q=copy.deepcopy(p);q['node'][:,15:,2:4]+=0.2;q['pair'][:,:,15:,:4]+=0.1
        np.testing.assert_array_equal(build(p,self.round_no)['values'][:,:15],build(q,self.round_no)['values'][:,:15])
    def test_summary_empty(self):self.assertIsNone(summarize([],self.round_no)[0]['rms'])
    def test_summary_support(self):
        f=build(sample(),self.round_no);s=summarize([f],self.round_no)
        self.assertEqual(s[0]['slots'],80);self.assertTrue(0<=s[0]['support']<=1)
    def test_unrecognized_round(self):
        with self.assertRaises(ValueError):build(sample(),True)
    def test_unrecognized_arm(self):
        with self.assertRaises(ValueError):view(build(sample(),self.round_no),'best')

class Goal(Shared):
    round_no=14
    def test_goal_axis_acceleration(self):
        p=sample();p['node'][...,4]=1;p['node'][...,5]=0
        a=build(p,14)
        # Player 2 in the fixture has dvx/dt=0.4 yards/s^2; goal axis is +x.
        np.testing.assert_allclose(a['values'][2,1:,0]*20,0.4,atol=1e-5)
        np.testing.assert_allclose(a['values'][2,1:,1],0,atol=1e-6)
    def test_zero_goal_masks_all(self):
        p=sample();p['node'][...,4:6]=0
        self.assertFalse(build(p,14)['valid'].any())
    def test_stationary_alignment_mask(self):
        p=sample();p['node'][...,2:4]=0
        self.assertFalse(build(p,14)['valid'][...,4:6].any())
    def test_no_gap_bridge(self):
        p=sample();p['node_valid'][2,8]=False
        f=build(p,14);self.assertFalse(f['valid'][2,8:10,:4].any())
    def test_jerk_requires_three_observations(self):
        self.assertFalse(build(sample(),14)['valid'][:,:2,2:4].any())

class Receiver(Shared):
    round_no=15
    def test_receiver_self_invalid(self):self.assertFalse(build(sample(),15)['valid'][0].any())
    def test_absent_receiver(self):
        p=sample();p['role'][0]=[0,1,0,0]
        self.assertFalse(build(p,15)['valid'].any())
    def test_relative_acceleration_units(self):
        a=build(sample(),15)
        # Receiver player 0 has acceleration 0; player 2 has acceleration +0.4.
        np.testing.assert_allclose(a['values'][2,1:,0]*20,-0.4,atol=1e-5)
    def test_joint_gap(self):
        p=sample();p['pair_valid'][2,0,8]=False
        a=build(p,15)
        self.assertFalse(a['valid'][2,8:10,:4].any())
        self.assertFalse(a['valid'][2,8:13,8:10].any())
    def test_jerk_has_three_point_support(self):
        self.assertFalse(build(sample(),15)['valid'][:,:2,10:12].any())

class Helpers(unittest.TestCase):
    def test_difference(self):
        x=np.arange(20)[None,:].astype(float);m=np.ones_like(x,bool)
        d,v=difference(x,m);np.testing.assert_allclose(d[:,1:],10)
        self.assertFalse(v[:,0].any())
    def test_difference_gap(self):
        x=np.arange(20)[None,:].astype(float);m=np.ones_like(x,bool);m[:,4]=False
        _,v=difference(x,m);self.assertFalse(v[:,4:6].any())
    def test_difference_bad_mask(self):
        with self.assertRaises(ValueError):difference(np.ones((1,20)),np.ones((1,20)))
if __name__=='__main__':unittest.main()
