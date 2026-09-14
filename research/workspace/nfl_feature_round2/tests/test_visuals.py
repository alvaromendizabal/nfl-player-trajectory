from pathlib import Path
import sys
import unittest
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import visuals

class VisualTests(unittest.TestCase):
    def test_actual_round1_charts_serialize(self):
        result=json.loads((Path(__file__).resolve().parents[1]/'evidence/round1_screen_summary.json').read_text())
        for function in (visuals.prior_scores,visuals.prior_fold_gains):
            fig=function(result);self.assertTrue(len(fig.data));json.loads(fig.to_json())
    def test_corrected_intervals_serialize(self):
        result={'phase':'attribution','decisions':[{'arm':'only_velocity','baseline':'control','delta_rmse':-.03,
            'simultaneous_low':-.05,'simultaneous_high':-.01,'direction':'addition','decision':'synthetic'}]}
        fig=visuals.contrast_intervals(result);self.assertTrue(len(fig.data));json.loads(fig.to_json())
    def test_training_support_charts(self):
        s={'training_only_support':[{'family':'turn','role':'Targeted Receiver','window':5,
            'available_fraction':.9,'players':10,'supported':9,'mean_adjacent_support':.8}]}
        for f in (visuals.support_bars,visuals.support_heatmap):self.assertTrue(len(f(s).data))
    def test_horizon_metric_uses_row_weighting(self):
        result={'slices':[{'arm':'arrival','fold':1,'slice':'first_second','rmse':1.,'rows':10,'sse':20.},
                           {'arm':'arrival','fold':2,'slice':'first_second','rmse':3.,'rows':30,'sse':540.}]}
        fig=visuals.horizon_scores(result);self.assertAlmostEqual(float(fig.data[0].y[0]),np.sqrt(7.))
