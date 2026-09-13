import copy,hashlib,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fixtures import model_sample
from families import build,ARMS
import model as m
from checkpoints import save_checkpoint,load_checkpoint,weight_hash
from study import metric,contrast,optimizer

def plays(r=12):
    result=[]
    for i in range(1,4):
        p=model_sample(i);e=build(p,r);p.update(extra=e['values'],extra_valid=e['valid']);result.append(p)
    return result

def trained(r=12):
    m.setup();p=plays(r);f=m.Forecast();opt=optimizer(f);norm=m.normalization(p)
    b=m.collate(p,'full',norm);y=torch.tensor(np.concatenate([x['y'] for x in p]),dtype=torch.float32)
    for _ in range(3):m.step(f,opt,b,y,2*len(y))
    return p,f,opt,norm,b,y

class Models(unittest.TestCase):
    def setUp(self):m.setup()
    def test_parameter_count(self):self.assertEqual(sum(p.numel() for p in m.Forecast().parameters()),19826)
    def test_identical_initializations(self):
        hs=[]
        for arm in ARMS:m.setup();hs.append(weight_hash(m.Forecast()))
        self.assertEqual(len(set(hs)),1)
    def test_equal_parameter_shapes_both_rounds(self):
        counts=[]
        for r in (12,13):
            p=plays(r);norm=m.normalization(p)
            for arm in ARMS:
                b=m.collate(p,arm,norm);self.assertEqual(b['node'].shape[-1],22);counts.append(sum(x.numel() for x in m.Forecast().parameters()))
        self.assertEqual(len(set(counts)),1)
    def test_exact_masks_all_arms(self):
        for r in (12,13):
            p=plays(r);norm=m.normalization(p);b=[m.collate(p,arm,norm) for arm in ARMS]
            for z in b[1:]:self.assertTrue(torch.equal(b[0]['node_valid'],z['node_valid']))
            self.assertTrue(torch.equal(b[0]['pair'],b[2]['pair']))
    def test_control_zero_values(self):
        p=plays();b=m.collate(p,'mask',m.normalization(p));self.assertFalse(b['node'][...,10:].any())
    def test_core_retains_exact_first_six(self):
        p=plays();norm=m.normalization(p);a=m.collate(p,'core',norm);b=m.collate(p,'full',norm)
        self.assertTrue(torch.equal(a['node'][...,:16],b['node'][...,:16]));self.assertFalse(a['node'][...,16:].any())
    def test_goal_values_zero_in_every_arm(self):
        p=plays();norm=m.normalization(p)
        for arm in ARMS:self.assertFalse(m.collate(p,arm,norm)['pair'][...,8:].any())
    def test_predictions_finite_shape(self):
        p=plays();f=m.Forecast();v=m.predict(f,p,'full',m.normalization(p));self.assertEqual(v.shape,(24,2));self.assertTrue(np.isfinite(v).all())
    def test_zero_initial_predictions(self):
        p=plays();np.testing.assert_array_equal(m.predict(m.Forecast(),p,'full',m.normalization(p)),0)
    def test_nonzero_feature_gradients(self):
        for r in (12,13):
            p,f,opt,norm,b,y=trained(r);f.zero_grad(set_to_none=True);b['node'].requires_grad_();f(**b).sum().backward()
            self.assertTrue(torch.isfinite(b['node'].grad).all());self.assertTrue(b['node'].grad[...,10:].abs().sum()>0)
    def test_prediction_replay_exact(self):
        p,f,opt,norm,b,y=trained()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);save_checkpoint(path,f,opt,3,'abc',[1.,2.,3.],'initial');g=m.Forecast();load_checkpoint(path,g,optimizer(g),'abc')
            np.testing.assert_array_equal(m.predict(f,p,'full',norm),m.predict(g,p,'full',norm))
    def test_interrupted_resume_exact(self):
        p=plays();norm=m.normalization(p);b=m.collate(p,'full',norm);y=torch.tensor(np.concatenate([x['y'] for x in p]),dtype=torch.float32)
        m.setup();clean=m.Forecast();co=optimizer(clean)
        for _ in range(6):m.step(clean,co,b,y,48)
        m.setup();f=m.Forecast();opt=optimizer(f)
        for _ in range(3):m.step(f,opt,b,y,48)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);save_checkpoint(path,f,opt,3,'abc',[1.,2.,3.],'initial');g=m.Forecast();go=optimizer(g);load_checkpoint(path,g,go,'abc')
            for _ in range(3):m.step(g,go,b,y,48)
            self.assertEqual(weight_hash(g),weight_hash(clean))
    def test_checkpoint_signature_rejected(self):
        p,f,opt,norm,b,y=trained()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);save_checkpoint(path,f,opt,3,'abc',[],'init')
            with self.assertRaises(ValueError):load_checkpoint(path,m.Forecast(),optimizer(m.Forecast()),'changed')
    def test_checkpoint_corruption_rejected(self):
        import json
        p,f,opt,norm,b,y=trained()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);save_checkpoint(path,f,opt,3,'abc',[],'init');r=json.loads((path/'checkpoint.json').read_text());(path/r['blob']).write_bytes(b'broken')
            with self.assertRaises(ValueError):load_checkpoint(path,m.Forecast(),opt,'abc')
    def test_no_model_mutation_predict(self):
        p,f,opt,norm,b,y=trained();before=weight_hash(f);m.predict(f,p,'full',norm);self.assertEqual(before,weight_hash(f))
    def test_input_immutable(self):
        p=plays();old=copy.deepcopy(p);m.collate(p,'core',m.normalization(p))
        for a,b in zip(p,old):
            for k in a:np.testing.assert_array_equal(a[k],b[k])
    def test_masked_poison_removed(self):
        p=plays();p[0]['extra_valid'][:]=False;p[0]['extra'][:]=np.nan;b=m.collate(p,'full',m.normalization(p));self.assertTrue(torch.isfinite(b['node']).all())
    def test_invalid_extra_rejected(self):
        p=plays();p[0]['extra'][1,4,0]=np.nan
        with self.assertRaises(ValueError):m.collate(p,'full',m.normalization(p))
    def test_metric_coordinate_factor(self):
        r=metric(np.zeros((2,2)),np.array([[3.,4.],[0,0]]));self.assertEqual(r['sse'],25);self.assertEqual(r['rmse'],2.5)
    def test_metric_shape_invalid(self):
        with self.assertRaises(ValueError):metric(np.ones((2,2)),np.ones((3,2)))
    def test_metric_nonfinite(self):
        with self.assertRaises(ValueError):metric(np.ones((2,2)),np.full((2,2),np.nan))
    def test_bootstrap_identical(self):
        y=np.zeros((4,2));a=np.ones((4,2));r=contrast(y,a,a,np.array([1,1,2,2]));self.assertEqual(r['high_adjusted'],0);self.assertFalse(r['passes'])
    def test_bootstrap_improvement(self):
        y=np.zeros((4,2));a=np.ones((4,2));r=contrast(y,a,a*.5,np.array([1,1,2,2]));self.assertTrue(r['passes']);self.assertEqual(r['planned_family_comparisons'],6)
    def test_bootstrap_failure(self):
        y=np.zeros((4,2));a=np.ones((4,2));r=contrast(y,a,a*2,np.array([1,1,2,2]));self.assertFalse(r['passes'])
    def test_bootstrap_one_game(self):
        with self.assertRaises(ValueError):contrast(np.zeros((4,2)),np.ones((4,2)),np.ones((4,2))*.5,np.ones(4))
    def test_shuffle_changes_no_comparison(self):
        y=np.zeros((4,2));a=np.ones((4,2));b=a*.5;g=np.array([1,1,2,2]);ix=[3,0,1,2]
        self.assertEqual(contrast(y,a,b,g),contrast(y[ix],a[ix],b[ix],g[ix]))
    def test_no_label_read_in_collate(self):
        class Guard(dict):
            def __getitem__(self,k):
                if k=='y':raise AssertionError('Target supplied to encoder')
                return super().__getitem__(k)
        p=plays();norm=m.normalization(p);m.collate([Guard(x) for x in p],'full',norm)
    def test_prediction_repeat(self):
        p,f,opt,norm,b,y=trained();np.testing.assert_array_equal(m.predict(f,p,'full',norm),m.predict(f,p,'full',norm))
