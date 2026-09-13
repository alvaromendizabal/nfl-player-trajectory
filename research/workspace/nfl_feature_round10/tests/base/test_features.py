import sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from fixtures import sample
from grouped_features import make_extension,pair_view
from goal_frame import build

class Features(unittest.TestCase):
    def setUp(self):
        self.p=sample();self.g=make_extension(self.p);self.p.update(self.g)
    def test_shape(self): self.assertEqual(self.g['goal'].shape,(4,4,20,6))
    def test_no_labels(self):
        self.p['y']=object();self.p['truth']=object();make_extension(self.p)
    def test_control_zeros_goal(self): self.assertTrue((pair_view(self.p,'cartesian')[0][...,8:]==0).all())
    def test_control_original_retained(self):np.testing.assert_array_equal(pair_view(self.p,'cartesian')[0][...,:8],self.p['pair'])
    def test_treatment_exact(self):np.testing.assert_array_equal(pair_view(self.p,'goal')[0][...,8:],self.g['goal'])
    def test_equal_masks(self):np.testing.assert_array_equal(pair_view(self.p,'goal')[1],pair_view(self.p,'cartesian')[1])
    def test_equal_ages(self):np.testing.assert_array_equal(pair_view(self.p,'goal')[2],pair_view(self.p,'cartesian')[2])
    def test_immutability(self):
        before={k:v.copy() for k,v in self.p.items()};make_extension(self.p);pair_view(self.p,'goal')
        for k,v in before.items():np.testing.assert_array_equal(self.p[k],v)
    def test_legacy_feature_parity(self):np.testing.assert_array_equal(self.g['goal'],build(self.p)['values'])
    def test_self_edges(self):self.assertFalse(self.g['goal_valid'][np.arange(4),np.arange(4)].any())
    def test_unknown_arm(self):
        with self.assertRaises(ValueError):pair_view(self.p,'winner')
    def test_available_poison(self):
        self.p['goal'][0,1,2,0]=np.inf
        with self.assertRaises(ValueError):pair_view(self.p,'goal')
    def test_missing_poison(self):
        self.p['goal_valid'][0,1,2,0]=False;self.p['goal'][0,1,2,0]=np.nan
        self.assertTrue(np.isfinite(pair_view(self.p,'goal')[0]).all())
    def test_bad_mask_dtype(self):
        self.p['goal_valid']=self.p['goal_valid'].astype(float)
        with self.assertRaises(ValueError):pair_view(self.p,'goal')
    def test_group_partition(self):
        g=self.g['groups'];self.assertFalse((g[...,0]&g[...,1]).any())
        np.testing.assert_array_equal(g[...,0]|g[...,1],self.p['pair_valid'][...,0])
    def test_goal_direction_undefined_mask(self):
        self.p['node'][0,:,4:6]=0;g=make_extension(self.p)
        self.assertFalse(g['goal_valid'][0,...,:4].any());self.assertTrue(np.isfinite(g['goal']).all())
    def test_finite(self):self.assertTrue(np.isfinite(self.g['goal']).all())
    def test_known_parallel_separation(self):
        self.p['node'][:,:,4]=1;self.p['node'][:,:,5]=0
        g=make_extension(self.p)
        np.testing.assert_allclose(g['goal'][0,1,:,0],self.p['pair'][0,1,:,0],atol=1e-7)
    def test_age_real_missing(self):
        self.p['pair_valid'][0,1,10:]=False
        g=make_extension(self.p);self.assertAlmostEqual(float(g['goal_age'][0,1,0]),1.0)
    def test_never_observed_age(self):
        self.p['pair_valid'][0,1]=False
        self.assertEqual(float(make_extension(self.p)['goal_age'][0,1,0]),2.0)
    def test_clock_gaps_preserved(self):
        self.p['pair_valid'][0,1,6]=False
        self.assertFalse(make_extension(self.p)['goal_valid'][0,1,6].any())
    def test_append_future_target_irrelevant(self):
        original=make_extension(self.p)['goal'];self.p['future_player_coordinates']=np.full((5,2),9999.)
        np.testing.assert_array_equal(original,make_extension(self.p)['goal'])
    def test_node_missing_poison(self):
        self.p['node_valid'][0,5]=False;self.p['node'][0,5]=np.nan
        self.assertTrue(np.isfinite(make_extension(self.p)['goal']).all())
    def test_different_view_values(self):self.assertGreater(np.count_nonzero(pair_view(self.p,'goal')[0]-pair_view(self.p,'cartesian')[0]),0)

if __name__=='__main__':unittest.main()
