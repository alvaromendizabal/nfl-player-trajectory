import copy,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fixtures import sample
import families as f
from new_features import build as core

class Families(unittest.TestCase):
    def test_two_complete_banks(self):
        for r in (12,13):
            self.assertEqual(len(f.NAMES[r]),12);self.assertEqual(len(set(f.NAMES[r])),12)
            self.assertEqual(len(f.UNITS[r]),12)
    def test_core_preserved_exact(self):
        p=sample();c=core(p)
        for r,sl in ((12,slice(0,6)),(13,slice(6,12))):
            a=f.build(p,r);np.testing.assert_array_equal(a['values'][...,:6],c['values'][...,sl]);np.testing.assert_array_equal(a['valid'][...,:6],c['valid'][...,sl])
    def test_shapes_types(self):
        for r in (12,13):
            a=f.build(sample(),r);self.assertEqual(a['values'].shape,(4,20,12));self.assertEqual(a['values'].dtype,np.float32);self.assertEqual(a['valid'].dtype,bool)
    def test_known_lateral_motion(self):
        p=sample();a=f.build(p,12);dp=p['pair'][:,0,:,:2]*20;dv=p['pair'][:,0,:,2:4]*10;dist=np.linalg.norm(dp,axis=-1)
        want=(dp[...,0]*dv[...,1]-dp[...,1]*dv[...,0])/np.maximum(dist,.1)
        np.testing.assert_allclose(a['values'][1:,:,6]*10,want[1:],atol=2e-6)
    def test_alignment(self):
        p=sample();np.testing.assert_array_equal(f.build(p,12)['values'][1:,:,7],p['pair'][1:,0,:,7])
    def test_receiver_self_mask(self):self.assertFalse(f.build(sample(),12)['valid'][0].any())
    def test_no_receiver(self):
        p=sample();p['role'][:]=[0,1,0,0];self.assertFalse(f.build(p,12)['valid'].any())
    def test_two_receivers(self):
        p=sample();p['role'][1]=[1,0,0,0]
        with self.assertRaises(ValueError):f.build(p,12)
    def test_translation_invariance(self):
        p=sample();q=copy.deepcopy(p);q['unused_absolute_xy']=np.full((4,2),123)
        for r in (12,13):np.testing.assert_array_equal(f.build(p,r)['values'],f.build(q,r)['values'])
    def test_permutation_equivariance(self):
        p=sample();ix=[3,0,2,1];q=copy.deepcopy(p)
        for k in ('node','node_valid','role'):q[k]=p[k][ix]
        for k in ('pair','pair_valid'):q[k]=p[k][ix][:,ix]
        for r in (12,13):np.testing.assert_array_equal(f.build(q,r)['values'],f.build(p,r)['values'][ix])
    def test_reflection(self):
        p=sample();q=copy.deepcopy(p);q['node'][...,[1,3,5,7]]*=-1;q['pair'][...,[1,3,6]]*=-1
        for r in (12,13):
            a=f.build(p,r);b=f.build(q,r);want=a['values'].copy();want[...,list(f.ODD[r])]*=-1
            np.testing.assert_allclose(want,b['values'],atol=1e-7);np.testing.assert_array_equal(a['valid'],b['valid'])
    def test_feature_causality(self):
        p=sample();q=copy.deepcopy(p);q['node'][:,15:,2:4]+=2;q['pair'][:,:,15:,:4]+=3
        for r in (12,13):np.testing.assert_array_equal(f.build(p,r)['values'][:,:15],f.build(q,r)['values'][:,:15])
    def test_no_targets_read(self):
        class Poison(dict):
            def __getitem__(self,key):
                if key in ('y','keys','query','base','train','goal'):raise AssertionError('Label/query access')
                return super().__getitem__(key)
        for r in (12,13):f.build(Poison(sample()),r)
    def test_missing_receiver_frame_does_not_bridge(self):
        p=sample();p['pair_valid'][:,0,8]=False
        a=f.build(p,12);self.assertFalse(a['valid'][:,8:10,8:10].any())
    def test_receiver_at_landing_masks_axis(self):
        p=sample();p['node'][...,4:6]=0;self.assertFalse(f.build(p,12)['valid'][...,10:12].any())
    def test_coincident_geometry(self):
        p=sample();p['pair'][...,:2]=0;a=f.build(p,12);self.assertFalse(a['valid'][...,6].any());self.assertFalse(a['valid'][...,8:12].any())
    def test_masked_poison_ignored(self):
        p=sample();p['node_valid'][1,5]=False;p['node'][1,5]=np.nan;p['pair'][0,0]=np.inf
        for r in (12,13):self.assertTrue(np.isfinite(f.build(p,r)['values']).all())
    def test_unmasked_poison_rejected(self):
        p=sample();p['node'][1,5,2]=np.nan
        for r in (12,13):
            with self.assertRaises(ValueError):f.build(p,r)
    def test_immutable(self):
        p=sample();old=copy.deepcopy(p)
        for x in p.values():x.flags.writeable=False
        for r in (12,13):f.build(p,r)
        for k in p:np.testing.assert_array_equal(p[k],old[k])
    def test_constant_acceleration_trailing(self):
        p=sample();a=f.build(p,13)
        np.testing.assert_allclose(a['values'][2,5:,[6,8]].T*20,.4,atol=7e-6)
    def test_no_window_bridging(self):
        p=sample();p['node_valid'][1,8]=False;a=f.build(p,13)
        self.assertFalse(a['valid'][1,8:12,6:8].any());self.assertFalse(a['valid'][1,8:14,8:10].any());self.assertFalse(a['valid'][1,8:11,10:12].any())
    def test_initial_window_support(self):
        a=f.build(sample(),13);self.assertFalse(a['valid'][:,:3,6:8].any());self.assertFalse(a['valid'][:,:5,8:10].any());self.assertFalse(a['valid'][:,:2,10:12].any())
    def test_constant_velocity_zero_derivatives(self):
        p=sample();p['node'][...,2]=.3;p['node'][...,3]=.2
        np.testing.assert_array_equal(f.build(p,13)['values'],0)
    def test_constant_acceleration_zero_jerk_tolerance(self):
        a=f.build(sample(),13);np.testing.assert_allclose(a['values'][:,2:,10:12]*100,0,atol=1e-4)
    def test_equal_masks_all_arms(self):
        for r in (12,13):
            p=f.build(sample(),r);views=[f.view(p,arm) for arm in f.ARMS]
            for v in views:np.testing.assert_array_equal(v[1],p['valid'])
            self.assertFalse(views[0][0].any());self.assertFalse(views[1][0][...,6:].any());np.testing.assert_array_equal(views[2][0],p['values'])
    def test_bad_arm(self):
        with self.assertRaises(ValueError):f.view(f.build(sample(),12),'unknown')
    def test_bad_round(self):
        for r in (True,11,14,'12'):
            with self.assertRaises(ValueError):f.build(sample(),r)
    def test_repeat_exact(self):
        for r in (12,13):np.testing.assert_array_equal(f.build(sample(),r)['values'],f.build(sample(),r)['values'])
    def test_summary_units_counts(self):
        for r in (12,13):
            a=f.build(sample(),r);s=f.summarize([a],r)
            self.assertEqual(len(s),12);self.assertEqual(s[0]['slots'],80);self.assertEqual(s[0]['valid'],int(a['valid'][...,0].sum()))
    def test_one_player_masks(self):
        a=f.build(sample(n=1),12);self.assertFalse(a['valid'].any())
    def test_zero_derivatives_are_not_missing(self):
        p=sample();p['node'][...,2:4]=.2;a=f.build(p,13)
        self.assertTrue(a['valid'][:,5:,6:10].all());self.assertFalse(a['values'][:,5:,6:10].any())
    def test_zero_distance_change_defined(self):
        p=sample();p['pair'][:,:,:,:2]=p['pair'][:,:,-1:,:2];a=f.build(p,12)
        self.assertTrue(a['valid'][1:,1:,9].all());self.assertFalse(a['values'][:,1:,9].any())
