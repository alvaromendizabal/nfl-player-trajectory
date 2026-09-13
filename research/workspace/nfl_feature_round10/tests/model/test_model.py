import unittest,sys,tempfile,copy,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from fixtures import model_sample
from grouped_model import Forecast,setup,collate,normalization,step,predict,ordered_batches
from checkpoints import save_checkpoint,load_checkpoint,weight_hash
from study import new_optimizer,rmse,contrast

class Model(unittest.TestCase):
    def setUp(self):
        setup();self.p=model_sample();self.norm=normalization([self.p]);self.m=Forecast()
        # Nonzero deterministic head for structural sensitivity tests only.
        torch.nn.init.normal_(self.m.decoder[-1].weight,std=.1)
    def test_finite(self):self.assertTrue(torch.isfinite(self.m(**collate([self.p],'goal',self.norm))).all())
    def test_shape(self):self.assertEqual(tuple(self.m(**collate([self.p],'goal',self.norm)).shape),(8,2))
    def test_equal_initialization(self):
        setup();a=Forecast();setup();b=Forecast();self.assertEqual(weight_hash(a),weight_hash(b))
    def test_same_masks(self):
        a=collate([self.p],'goal',self.norm);b=collate([self.p],'cartesian',self.norm)
        for k in a:
            if k!='pair':self.assertTrue(torch.equal(a[k],b[k]),k)
        self.assertTrue(torch.equal(a['pair'][...,:8],b['pair'][...,:8]))
    def test_value_sensitivity(self):
        a=self.m(**collate([self.p],'goal',self.norm));b=self.m(**collate([self.p],'cartesian',self.norm))
        self.assertGreater(float((a-b).abs().max().detach()),0)
    def test_goal_gradient(self):
        b=collate([self.p],'goal',self.norm);b['pair'].requires_grad_();self.m(**b).sum().backward()
        self.assertGreater(float(b['pair'].grad[...,8:].abs().max()),0)
    def test_context_sensitivity(self):
        b=collate([self.p],'goal',self.norm)
        self.assertGreater(float((self.m(**b)-self.m(**b,zero_context=True)).abs().max().detach()),0)
    def test_immutable(self):
        old={k:v.copy() for k,v in self.p.items()};predict(self.m,[self.p],'goal',self.norm)
        for k,v in old.items():np.testing.assert_array_equal(self.p[k],v)
    def test_no_target_input(self):
        a=collate([self.p],'goal',self.norm);self.p['y']=object();b=collate([self.p],'goal',self.norm)
        for k in a:self.assertTrue(torch.equal(a[k],b[k]))
    def test_query_permutation(self):
        ix=np.arange(8)[::-1];q=copy.deepcopy(self.p)
        for key in ('base','query','keys','y','role_query'):q[key]=q[key][ix].copy()
        np.testing.assert_allclose(predict(self.m,[q],'goal',self.norm),predict(self.m,[self.p],'goal',self.norm)[ix],atol=2e-7)
    def test_player_permutation(self):
        q=copy.deepcopy(self.p);ix=np.array([2,0,3,1]);inv=np.argsort(ix)
        for k in ('node','node_valid','node_age','role','side','ids'):q[k]=q[k][ix]
        for k in ('pair','pair_valid','pair_age','goal','goal_valid','goal_age','groups'):q[k]=q[k][ix][:,ix]
        q['query']=inv[q['query']]
        np.testing.assert_allclose(predict(self.m,[q],'goal',self.norm),predict(self.m,[self.p],'goal',self.norm),atol=2e-7)
    def test_padding(self):
        larger=model_sample(2,6)
        a=predict(self.m,[self.p],'goal',self.norm)
        b=predict(self.m,[self.p,larger],'goal',self.norm)[:8]
        np.testing.assert_allclose(a,b,atol=3e-7)
    def test_no_peer(self):
        p=model_sample(1,1)
        self.assertTrue(np.isfinite(predict(self.m,[p],'goal',self.norm)).all())
    def test_gradient_finite(self):
        b=collate([self.p],'goal',self.norm);loss=self.m(**b).square().mean();loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.m.parameters() if p.grad is not None))
    def test_step_changes_weights(self):
        old=weight_hash(self.m);step(self.m,new_optimizer(self.m),collate([self.p],'goal',self.norm),torch.tensor(self.p['y'],dtype=torch.float32),16)
        self.assertNotEqual(old,weight_hash(self.m))
    def test_invalid_query(self):
        self.p['query'][0]=99
        with self.assertRaises(ValueError):collate([self.p],'goal',self.norm)
    def test_checkpoint_continuation(self):
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t);setup();a=Forecast();opt=new_optimizer(a);b=collate([self.p],'goal',self.norm);y=torch.tensor(self.p['y'],dtype=torch.float32)
            for _ in range(3):step(a,opt,b,y,16)
            save_checkpoint(folder,a,opt,3,'fixture',[1.,2.,3.],'initial')
            c=Forecast();co=new_optimizer(c);state=load_checkpoint(folder,c,co,'fixture')
            self.assertEqual(state['step'],3)
            for _ in range(3):step(a,opt,b,y,16);step(c,co,b,y,16)
            self.assertEqual(weight_hash(a),weight_hash(c))
            np.testing.assert_array_equal(predict(a,[self.p],'goal',self.norm),predict(c,[self.p],'goal',self.norm))
    def test_corrupt_checkpoint_reject(self):
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t);save_checkpoint(folder,self.m,new_optimizer(self.m),0,'fixture',[],'initial')
            r=json.loads((folder/'checkpoint.json').read_text());(folder/r['blob']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):load_checkpoint(folder,Forecast(),new_optimizer(Forecast()),'fixture')
    def test_batch_order(self):
        for x,y in zip(ordered_batches(20,3),ordered_batches(20,3)):np.testing.assert_array_equal(x,y)
    def test_rmse_formula(self):self.assertEqual(rmse(np.zeros((1,2)),np.array([[3.,4.]])),np.sqrt(12.5))
    def test_rmse_bad_shape(self):
        with self.assertRaises(ValueError):rmse(np.zeros((1,2)),np.zeros((2,2)))
    def test_bootstrap(self):
        y=np.ones((20,2));c=np.zeros_like(y);p=np.ones_like(y)*.5;g=np.repeat(np.arange(4),5)
        r=contrast(y,c,p,g);self.assertTrue(r['feature_gate_passed']);self.assertEqual(r['relative_gain'],.5)
    def test_bootstrap_one_game_reject(self):
        with self.assertRaises(ValueError):contrast(np.ones((3,2)),np.zeros((3,2)),np.zeros((3,2)),np.ones(3))
    def test_eval_norm_not_used(self):
        norm=normalization([self.p]);other=model_sample(99);other['base'][:]=1e9
        self.assertEqual(norm,normalization([self.p]))

if __name__=='__main__':unittest.main()
