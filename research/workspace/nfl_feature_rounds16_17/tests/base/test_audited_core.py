import unittest
import copy
import numpy as np
from new_features import build,NAMES,SCALES,ODD,summarize


def sample(n=3):
    t=np.arange(20)*.1
    node=np.zeros((n,20,10),np.float32);nv=np.ones_like(node,bool)
    vel=np.zeros((n,20,2));pos=np.zeros_like(vel)
    for i in range(n):
        vel[i,:,0]=2+i+2*t;vel[i,:,1]=i*.5
        pos[i,:,0]=i*4+(2+i)*t+t*t;pos[i,:,1]=i*3+i*.5*t
    node[...,:2]=(pos-pos[:,-1:, :])/10;node[...,2:4]=vel/10
    node[...,4]=.4;node[...,5]=.5;node[...,6]=1;node[...,8]=.5
    pair=np.zeros((n,n,20,8),np.float32)
    dp=pos[None]-pos[:,None];dv=vel[None]-vel[:,None];d=np.linalg.norm(dp,axis=-1)
    pair[...,:2]=dp/20;pair[...,2:4]=dv/10;pair[...,4]=d/20
    pair[...,5]=-(dp*dv).sum(-1)/np.maximum(d,.1)/10
    valid=np.broadcast_to(~np.eye(n,dtype=bool)[...,None,None],pair.shape).copy()
    role=np.zeros((n,4),np.float32);role[:,1]=1;role[0]=[1,0,0,0]
    return {'node':node,'node_valid':nv,'pair':pair,'pair_valid':valid,'role':role}

class Tests(unittest.TestCase):
    def test_shape(self):self.assertEqual(build(sample())['values'].shape,(3,20,12))
    def test_dtype(self):self.assertEqual(build(sample())['values'].dtype,np.float32)
    def test_names(self):self.assertEqual(len(NAMES),len(set(NAMES)))
    def test_finite(self):self.assertTrue(np.isfinite(build(sample())['values']).all())
    def test_known_acceleration(self):
        a=build(sample());np.testing.assert_allclose(a['values'][0,1:,6]*20,2,atol=4e-6)
    def test_receiver_geometry(self):
        p=sample();r=build(p);np.testing.assert_array_equal(r['values'][1,:,:6],p['pair'][1,0,:,:6])
    def test_receiver_self_mask(self):self.assertFalse(build(sample())['valid'][0,:,:6].any())
    def test_first_difference_missing(self):self.assertFalse(build(sample())['valid'][:,0,6:].any())
    def test_missing_receiver(self):
        p=sample();p['role'][:]=[0,1,0,0];self.assertFalse(build(p)['valid'][...,:6].any())
    def test_two_receivers_rejected(self):
        p=sample();p['role'][1]=[1,0,0,0]
        with self.assertRaises(ValueError):build(p)
    def test_missing_frame(self):
        p=sample();p['node_valid'][1,9]=False
        r=build(p);self.assertFalse(r['valid'][1,9:11,6:].any());self.assertTrue(r['valid'][1,11,6])
    def test_no_gap_bridge(self):
        p=sample();p['node_valid'][1,4:9]=False
        self.assertFalse(build(p)['valid'][1,4:10,6:].any())
    def test_invalid_velocity(self):
        p=sample();p['node_valid'][1,8,2]=False;self.assertFalse(build(p)['valid'][1,8:10,6:].any())
    def test_receiver_mask(self):
        p=sample();p['pair_valid'][2,0,5,:6]=False;self.assertFalse(build(p)['valid'][2,5,:6].any())
    def test_masked_poison(self):
        p=sample();p['node_valid'][1,8]=False;p['node'][1,8]=np.nan;p['pair'][0,0]=np.inf
        self.assertTrue(np.isfinite(build(p)['values']).all())
    def test_unmasked_poison(self):
        p=sample();p['node'][0,1,2]=np.nan
        with self.assertRaises(ValueError):build(p)
    def test_stationary(self):
        p=sample();p['node'][...,2:4]=0;r=build(p)
        self.assertFalse(r['valid'][...,8:11].any());self.assertTrue(r['valid'][:,1:,11].all())
    def test_no_wrap_spike(self):
        p=sample();theta=np.deg2rad(np.linspace(179,181,20));p['node'][...,2]=np.cos(theta);p['node'][...,3]=np.sin(theta)
        np.testing.assert_allclose(build(p)['values'][:,1:,10]*10,np.deg2rad(2/19)/.1,atol=2e-6)
    def test_reflection(self):
        p=sample();q=copy.deepcopy(p);q['node'][...,[1,3,5,7]]*=-1;q['pair'][...,[1,3,6]]*=-1
        a=build(p);b=build(q);want=a['values'].copy();want[...,list(ODD)]*=-1
        np.testing.assert_allclose(want,b['values'],atol=1e-7);np.testing.assert_array_equal(a['valid'],b['valid'])
    def test_permutation(self):
        p=sample();ix=[2,0,1];q={k:(x[ix][:,ix] if k.startswith('pair') else x[ix]) for k,x in p.items()}
        np.testing.assert_array_equal(build(q)['values'],build(p)['values'][ix])
    def test_immutable(self):
        p=sample();old=copy.deepcopy(p)
        for x in p.values():x.flags.writeable=False
        build(p)
        for k in p:np.testing.assert_array_equal(p[k],old[k])
    def test_future_slots_no_effect_on_past(self):
        p=sample();q=copy.deepcopy(p);q['node'][:,15:,2:4]+=2
        np.testing.assert_array_equal(build(p)['values'][:,:15],build(q)['values'][:,:15])
    def test_labels_not_read(self):
        class Poison(dict):
            def __getitem__(self,k):
                if k in ['y','keys','base','goal','query','train']:raise AssertionError('Forbidden field read')
                return super().__getitem__(k)
        build(Poison(sample()))
    def test_self_pairs_rejected(self):
        p=sample();p['pair_valid'][0,0,0,0]=True
        with self.assertRaises(ValueError):build(p)
    def test_bad_role(self):
        p=sample();p['role'][0]=[.5,.5,0,0]
        with self.assertRaises(ValueError):build(p)
    def test_bad_mask(self):
        p=sample();p['node_valid']=p['node_valid'].astype(int)
        with self.assertRaises(ValueError):build(p)
    def test_summary(self):
        p=build(sample());s=summarize([p]);self.assertEqual(len(s),12);self.assertEqual(s[0]['valid'],40)
    def test_summary_empty(self):self.assertEqual(summarize([])[0]['valid'],0)
    def test_summary_missing_values(self):
        p=build(sample());p['valid'][:]=False;p['values'][:]=0
        self.assertIsNone(summarize([p])[0]['rms'])
    def test_translation(self):
        # Original inputs are already relative; changing unused absolute anchors cannot affect output.
        p=sample();p['xy']=np.zeros((3,2));a=build(p);p['xy']+=100
        np.testing.assert_array_equal(a['values'],build(p)['values'])
    def test_repeat_exact(self):np.testing.assert_array_equal(build(sample())['values'],build(sample())['values'])
    def test_reject_player_count(self):
        p=sample(23)
        with self.assertRaises(ValueError):build(p)
    def test_speeds_increasing(self):
        s=build(sample());self.assertTrue((s['values'][:,1:,11]>0).all())
