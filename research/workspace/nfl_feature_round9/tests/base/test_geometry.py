import unittest
import sys
from pathlib import Path
import copy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fixtures import sample
from goal_frame import build,NAMES,SCALES,GROUPS

class GeometryTests(unittest.TestCase):
    def setUp(self):self.p=sample()
    def test_shape(self):self.assertEqual(build(self.p)['values'].shape,(4,4,20,6))
    def test_finite(self):self.assertTrue(np.isfinite(build(self.p)['values']).all())
    def test_names_unique(self):self.assertEqual(len(set(NAMES)),6)
    def test_group_names(self):self.assertEqual(len(GROUPS),4)
    def test_self_mask(self):self.assertFalse(build(self.p)['valid'][np.arange(4),np.arange(4)].any())
    def test_self_groups(self):self.assertFalse(build(self.p)['groups'][np.arange(4),np.arange(4)].any())
    def test_units_projection(self):
        f=build(self.p);b=self.p['node'][0,0,4:6].astype(float)*20;u=b/np.linalg.norm(b)
        delta=self.p['pair'][0,1,0,:2].astype(float)*20
        self.assertAlmostEqual(float(f['values'][0,1,0,0]*20),float(delta@u),places=6)
    def test_inverse_projection(self):
        f=build(self.p);b=self.p['node'][...,4:6].astype(float)*20;u=b/np.linalg.norm(b,axis=-1)[...,None]
        a=f['values'][...,0:2]*20
        back=a[...,0,None]*u[:,None]+a[...,1,None]*np.stack([-u[...,1],u[...,0]],-1)[:,None]
        np.testing.assert_allclose(back,self.p['pair'][...,:2]*20,atol=2e-6,rtol=0)
    def test_reflection(self):
        p=copy.deepcopy(self.p);p['node'][...,[1,3,5,7]]*=-1;p['pair'][...,[1,3,6]]*=-1
        a=build(self.p);b=build(p);want=a['values'].copy();want[...,[1,3]]*=-1
        np.testing.assert_array_equal(want,b['values'])
    def test_permutation(self):
        q=np.array([2,0,3,1]);p=copy.deepcopy(self.p)
        for k in ('node','node_valid','role','side'):p[k]=p[k][q]
        for k in ('pair','pair_valid'):p[k]=p[k][q][:,q]
        a=build(self.p);b=build(p)
        for k in a:np.testing.assert_array_equal(a[k][q][:,q],b[k])
    def test_gap(self):
        self.p['pair_valid'][:,:,5]=False
        f=build(self.p);self.assertFalse(f['valid'][:,:,5].any());self.assertFalse(f['groups'][:,:,5].any())
    def test_gap_poison(self):
        self.p['pair_valid'][:,:,5]=False;self.p['pair'][:,:,5]=np.nan
        self.assertTrue(np.isfinite(build(self.p)['values']).all())
    def test_node_poison_hidden(self):
        self.p['node_valid'][0,5]=False;self.p['node'][0,5]=np.nan
        self.assertTrue(np.isfinite(build(self.p)['values']).all())
    def test_observed_poison_rejected(self):
        self.p['pair'][0,1,0,0]=np.nan
        with self.assertRaises(ValueError):build(self.p)
    def test_zero_goal_axis_mask(self):
        self.p['node'][0,:,4:6]=0
        self.assertFalse(build(self.p)['valid'][0,:,:,:4].any())
    def test_velocity_unavailable(self):
        self.p['pair_valid'][...,2:4]=False
        self.assertFalse(build(self.p)['valid'][...,2:4].any())
    def test_side_groups_partition(self):
        g=build(self.p)['groups'];joint=self.p['pair_valid'][...,0]
        np.testing.assert_array_equal(g[...,0]|g[...,1],joint)
        self.assertFalse((g[...,0]&g[...,1]).any())
    def test_passer_index_is_two(self):
        g=build(self.p)['groups'];self.assertTrue(g[0,2,:,3].all());self.assertFalse(g[:,3,:,3].any())
    def test_receiver_index_zero(self):
        g=build(self.p)['groups'];self.assertTrue(g[1,0,:,2].all());self.assertFalse(g[:,1,:,2].any())
    def test_targets_ignored(self):
        old=build(self.p);self.p['y']=np.ones((8,2))*9999;self.p['truth']='poison'
        for k,v in old.items():np.testing.assert_array_equal(v,build(self.p)[k])
    def test_query_ignored(self):
        old=build(self.p);self.p['query']=np.array([-999])
        for k,v in old.items():np.testing.assert_array_equal(v,build(self.p)[k])
    def test_immutable(self):
        old={k:v.copy() for k,v in self.p.items()};build(self.p)
        for k in old:np.testing.assert_array_equal(old[k],self.p[k])
    def test_bad_role(self):
        self.p['role'][0]=0
        with self.assertRaises(ValueError):build(self.p)
    def test_bad_shape(self):
        self.p['pair']=self.p['pair'][:,:,:,:7]
        with self.assertRaises(ValueError):build(self.p)
    def test_bad_boolean(self):
        self.p['node_valid']=self.p['node_valid'].astype(int)
        with self.assertRaises(ValueError):build(self.p)
    def test_future_slots_do_not_change_earlier(self):
        old=build(self.p);self.p['node'][:,12:]+=1;self.p['pair'][:,:,12:]+=1
        for k,v in old.items():np.testing.assert_array_equal(v[:,:,:12],build(self.p)[k][:,:,:12])
    def test_22_players_bounded(self):
        f=build(sample(n=22));self.assertLess(sum(v.nbytes for v in f.values()),400000)

if __name__=='__main__':unittest.main()
