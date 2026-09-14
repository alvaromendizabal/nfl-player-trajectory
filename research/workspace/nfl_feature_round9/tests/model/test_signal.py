import unittest
import sys
import copy
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fixtures import sample
from signal_probe import reverse_valid,probe,input_contrast,weights_digest,prediction_with_branch_removed
from sequence_model import Forecast,setup,collate
from sequence_features import terminal_view

class SignalTests(unittest.TestCase):
    def setUp(self):
        setup();self.model=Forecast();self.p=sample();self.norm={'mean':[0.]*72,'scale':[1.]*72}
    def enable(self):
        with torch.no_grad():
            self.model.decoder[-1].weight.normal_(0,.1)
    def test_zero_head_no_sensitivity(self):
        r=probe(self.model,self.p,'history',self.norm,collate)
        self.assertEqual(sum(r['gradient_energy']),0)
        self.assertEqual(sum(x['sum_squared_change'] for x in r['changes'].values()),0)
    def test_enabled_pair_gradient(self):
        self.enable();r=probe(self.model,self.p,'history',self.norm,collate)
        self.assertGreater(sum(r['gradient_energy']),0)
    def test_enabled_pair_effect(self):
        self.enable();r=probe(self.model,self.p,'history',self.norm,collate)
        self.assertGreater(r['changes']['swap_pair_view']['sum_squared_change'],0)
    def test_weights_unchanged(self):
        self.enable();h=weights_digest(self.model);probe(self.model,self.p,'history',self.norm,collate)
        self.assertEqual(h,weights_digest(self.model))
    def test_no_parameter_gradients(self):
        self.enable();probe(self.model,self.p,'history',self.norm,collate)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
    def test_input_unchanged(self):
        self.enable();old=copy.deepcopy(self.p);probe(self.model,self.p,'history',self.norm,collate)
        for k in old:np.testing.assert_array_equal(self.p[k],old[k])
    def test_repeat_exact(self):
        self.enable();a=probe(self.model,self.p,'history',self.norm,collate);b=probe(self.model,self.p,'history',self.norm,collate)
        self.assertEqual(a,b)
    def test_reverse_involution(self):
        b=collate([self.p],'history',self.norm);x=b['pair'];v=b['pair_valid']
        self.assertTrue(torch.equal(reverse_valid(reverse_valid(x,v),v),x))
    def test_reverse_keeps_invalid(self):
        b=collate([self.p],'history',self.norm);v=b['pair_valid'];r=reverse_valid(b['pair'],v)
        self.assertTrue(torch.equal(r[~v],b['pair'][~v]))
    def test_terminal_reversal_unchanged(self):
        b=collate([self.p],'terminal',self.norm)
        self.assertTrue(torch.equal(b['pair'],reverse_valid(b['pair'],b['pair_valid'])))
    def test_input_contrast_nonzero(self):self.assertGreater(sum(input_contrast(self.p,terminal_view)['sum_squared_difference']),0)
    def test_target_poison_ignored(self):
        self.enable();a=probe(self.model,self.p,'history',self.norm,collate);self.p['y']='DO NOT READ'
        self.assertEqual(a,probe(self.model,self.p,'history',self.norm,collate))
    def test_hook_removed(self):
        self.enable();b=collate([self.p],'history',self.norm);prediction_with_branch_removed(self.model,b,'context')
        self.assertEqual(len(self.model.decoder[0]._forward_pre_hooks),0)
    def test_unknown_branch(self):
        b=collate([self.p],'history',self.norm)
        with self.assertRaises(ValueError):prediction_with_branch_removed(self.model,b,'anything')

if __name__=='__main__':unittest.main()
