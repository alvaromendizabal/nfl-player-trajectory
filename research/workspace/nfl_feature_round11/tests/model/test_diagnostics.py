import unittest
import numpy as np
import torch
from diagnostics import aggregate
from grouped_model import Forecast,setup

class Tests(unittest.TestCase):
    def test_known_rmse(self):
        m,s=aggregate(np.zeros((2,2)),np.array([[1,1],[3,3.]]),np.array([1,12]),np.array([0,1]))
        self.assertAlmostEqual(m['rmse'],np.sqrt(5));self.assertEqual(m['sse'],20)
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):aggregate(np.zeros((0,2)),np.zeros((0,2)),np.array([]),np.array([]))
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):aggregate(np.zeros((1,2)),np.array([[np.nan,0]]),np.array([1]),np.array([0]))
    def test_squared_error_accounting(self):
        y=np.arange(30).reshape(15,2);m,s=aggregate(y,y*.7,np.arange(1,16),np.zeros(15,int))
        self.assertAlmostEqual(m['sse'],sum(x['sse'] for x in s if 'second' in x['slice']))
    def test_shape_rejected(self):
        with self.assertRaises(ValueError):aggregate(np.zeros((1,2)),np.zeros((2,2)),np.array([1]),np.array([0]))
    def test_roles(self):
        m,s=aggregate(np.zeros((3,2)),np.ones((3,2)),np.arange(1,4),np.array([0,1,1]))
        self.assertEqual([x['rows'] for x in s if x['slice']=='role_1'],[2])
    def test_original_model_size(self):self.assertEqual(sum(x.numel() for x in Forecast().parameters()),19250)
    def test_no_optimizer_in_audit(self):
        import inspect,diagnostics
        src=inspect.getsource(diagnostics.run_audit)
        import ast
        tree=ast.parse(src)
        forbidden={'step','backward','zero_grad','Adam','AdamW','SGD'}
        calls=[n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else '' for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(forbidden.intersection(calls))
