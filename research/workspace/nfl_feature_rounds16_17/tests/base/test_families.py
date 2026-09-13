"""User-executed analytic regression tests. No test has been run by the assistant."""
import copy,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fixtures import sample
from families import build,view,NAMES,SCALES,ODD,summarize

class Common:
    def test_shape(self):
        f=build(sample(),self.r);self.assertEqual(f['values'].shape,(4,20,12));self.assertEqual(f['valid'].dtype,np.bool_)
    def test_float32(self):self.assertEqual(build(sample(),self.r)['values'].dtype,np.float32)
    def test_unique_names(self):self.assertEqual(len(set(NAMES[self.r])),12)
    def test_finite(self):self.assertTrue(np.isfinite(build(sample(),self.r)['values']).all())
    def test_repeat(self):
        a,b=build(sample(),self.r),build(sample(),self.r)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
    def test_immutable(self):
        p=sample();q=copy.deepcopy(p);build(p,self.r)
        for k in p:np.testing.assert_array_equal(p[k],q[k])
    def test_read_only(self):
        p=sample()
        for x in p.values():x.flags.writeable=False
        self.assertTrue(np.isfinite(build(p,self.r)['values']).all())
    def test_hidden_poison(self):
        p=sample();q=copy.deepcopy(p)
        p['node_valid'][2,8]=False;q['node_valid'][2,8]=False
        p['pair_valid'][:,:,8]=False;q['pair_valid'][:,:,8]=False
        q['node'][2,8]=np.nan;q['pair'][:,:,8]=np.inf
        a,b=build(p,self.r),build(q,self.r)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
    def test_supported_nan_rejected(self):
        p=sample();p['node'][1,8,0]=np.nan
        with self.assertRaises(ValueError):build(p,self.r)
    def test_target_excluded(self):
        class Guard(dict):
            def __getitem__(self,k):
                if k in ('y','keys','truth','pred'):raise AssertionError('Target access')
                return super().__getitem__(k)
        a=build(sample(),self.r);b=build(Guard(sample()),self.r)
        np.testing.assert_array_equal(a['values'],b['values'])
    def test_no_future_observation(self):
        p=sample();q=copy.deepcopy(p);q['node'][:,15:]+=1;q['pair'][:,:,15:]+=1
        np.testing.assert_array_equal(build(p,self.r)['values'][:,:15],build(q,self.r)['values'][:,:15])
    def test_masks_same_in_arms(self):
        f=build(sample(),self.r);a=[view(f,k) for k in ('mask','core','full')]
        for _,m in a:np.testing.assert_array_equal(m,f['valid'])
        self.assertFalse(a[0][0].any());self.assertFalse(a[1][0][...,6:].any())
        np.testing.assert_array_equal(a[1][0][...,:6],a[2][0][...,:6])
    def test_permutation(self):
        p=sample();q=copy.deepcopy(p);ix=np.array([2,0,3,1])
        for k in ('node','node_valid','role','side','ids','node_age'):q[k]=q[k][ix]
        for k in ('pair','pair_valid','pair_age'):q[k]=q[k][ix][:,ix]
        a,b=build(p,self.r),build(q,self.r)
        np.testing.assert_allclose(b['values'],a['values'][ix],atol=1e-6,rtol=0)
        np.testing.assert_array_equal(b['valid'],a['valid'][ix])
    def test_reflection(self):
        p=sample();q=copy.deepcopy(p);q['node'][...,[1,3,5,7]]*=-1;q['pair'][...,[1,3,6]]*=-1
        a,b=build(p,self.r),build(q,self.r);want=a['values'].copy();want[...,list(ODD[self.r])]*=-1
        np.testing.assert_allclose(want,b['values'],atol=1e-6,rtol=0)
        np.testing.assert_array_equal(a['valid'],b['valid'])
    def test_summary_empty(self):self.assertIsNone(summarize([],self.r)[0]['rms'])
    def test_summary_support(self):
        s=summarize([build(sample(),self.r)],self.r)
        self.assertEqual(s[0]['slots'],80);self.assertTrue(all(0<=x['support']<=1 for x in s))
    def test_bad_arm(self):
        with self.assertRaises(ValueError):view(build(sample(),self.r),'winner')
    def test_bad_round(self):
        with self.assertRaises(ValueError):build(sample(),True)

class Routes(Common,unittest.TestCase):
    r=16
    def straight(self):
        p=sample();p['node'][...,:2]=0;p['node'][...,0]=np.arange(20)[None,:]*.1/10
        p['node'][...,2:4]=0;p['node'][...,2]=.1
        return p
    def test_straight_units(self):
        f=build(self.straight(),16)
        np.testing.assert_allclose(f['values'][:,2:,0]*10,.2,rtol=0,atol=1e-6)
        np.testing.assert_allclose(f['values'][:,4:,3]*10,.4,rtol=0,atol=1e-6)
        np.testing.assert_allclose(f['values'][:,9:,6]*10,.9,rtol=0,atol=1e-6)
    def test_straight_efficiency(self):
        f=build(self.straight(),16)
        for j in (2,5,8):np.testing.assert_allclose(f['values'][...,j][f['valid'][...,j]],1,atol=1e-6)
    def test_turns_zero_straight(self):
        f=build(self.straight(),16);np.testing.assert_allclose(f['values'][:,9:,9:11],0,atol=1e-6)
        np.testing.assert_allclose(f['values'][:,9:,11],1,atol=1e-6)
    def test_gaps_not_bridged(self):
        p=self.straight();p['node_valid'][2,8,:2]=False;f=build(p,16)
        self.assertFalse(f['valid'][2,8:18,6:].any())
    def test_window_initial_support(self):
        f=build(sample(),16);self.assertFalse(f['valid'][:,:2,:3].any());self.assertFalse(f['valid'][:,:9,6:].any())
    def test_zero_path_invalid(self):
        p=self.straight();p['node'][...,:2]=0;f=build(p,16)
        for j in (2,5,8,9,10,11):self.assertFalse(f['valid'][...,j].any())
    def test_zero_heading_does_not_invent_orientation(self):
        p=self.straight();p['node'][...,2:4]=0;f=build(p,16)
        self.assertFalse(f['valid'][...,[0,1,3,4,6,7]].any())
    def test_translation_cancels(self):
        p=self.straight();q=copy.deepcopy(p);q['node'][...,:2]+=[3,4]
        a,b=build(p,16),build(q,16);np.testing.assert_allclose(a['values'],b['values'],atol=2e-5,rtol=0)

class Traffic(Common,unittest.TestCase):
    r=17
    def one(self):
        p=sample();p['pair_valid'][:]=False;p['pair'][:]=0
        p['pair_valid'][2,3]=True;p['pair'][2,3,:,0]=2/20;p['pair'][2,3,:,2]=-1/10
        return p
    def test_one_opponent_physics(self):
        f=build(self.one(),17);x=f['values'][2]*SCALES[17]
        np.testing.assert_allclose(x[:,:6],np.tile([1,1,2,1,1,1],(20,1)),atol=1e-6,rtol=0)
    def test_no_observed_opponents_masked(self):
        p=sample();p['side'][:]=0;f=build(p,17);self.assertFalse(f['valid'].any());self.assertFalse(f['values'].any())
    def test_no_self_pairs(self):
        p=sample();p['pair_valid'][0,0]=True
        with self.assertRaises(ValueError):build(p,17)
    def test_stationary_relative_time_zero(self):
        p=self.one();p['pair'][...,2:4]=0;f=build(p,17)
        np.testing.assert_array_equal(f['values'][...,5],0)
    def test_absent_goal_corridor_masked(self):
        p=self.one();p['node_valid'][...,4:6]=False;f=build(p,17)
        self.assertFalse(f['valid'][...,10].any());self.assertTrue(f['valid'][2,:,:6].any())
    def test_current_frame_gap_not_filled(self):
        p=self.one();p['pair_valid'][2,3,8]=False;f=build(p,17)
        self.assertFalse(f['valid'][2,8].any());self.assertTrue(f['valid'][2,9,:6].all())
    def test_hypothetical_time_bounded(self):
        f=build(sample(),17);x=f['values'][...,5][f['valid'][...,5]]
        self.assertTrue(((x>=0)&(x<=1)).all())
    def test_distance_never_negative(self):
        f=build(sample(),17)
        for j in (2,4):self.assertTrue((f['values'][...,j]>=0).all())

if __name__=='__main__':unittest.main()
