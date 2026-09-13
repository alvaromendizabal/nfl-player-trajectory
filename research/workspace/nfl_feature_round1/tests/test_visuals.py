from __future__ import annotations
import json
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import visuals
from origin_features import ALL_NAMES,make_example
from test_origin_features import fixture

class PlotTests(unittest.TestCase):
    def test_origins_chart_existing_rows_column(self):
        s={'offsets':{'0':{'rows':20,'prethrow_rows':0},'5':{'rows':30,'prethrow_rows':10}}}
        fig=visuals.origins_figure(s);self.assertEqual(len(fig.data),2);self.assertTrue(fig.to_json())
    def test_old_error_mass_uses_row_weighted_squares(self):
        p=Path(__file__).resolve().parents[1]/'evidence/previous_workspace_receipts.json'
        frame=visuals.old_error_slices(json.loads(p.read_text()))
        self.assertAlmostEqual(frame.loc[frame.value>1,'squared_error_share'].sum(),.8357297110604606)
        self.assertTrue(visuals.error_mass_figure(frame).to_json())
    def test_origin_horizon_and_activation_charts(self):
        raw,y=fixture();e=make_example(raw,y,5)
        self.assertTrue(visuals.horizon_figure(e).to_json())
        fig,support=visuals.feature_support_figure(e,ALL_NAMES)
        self.assertTrue(fig.to_json());self.assertEqual(len(support),len(ALL_NAMES))
    def test_bootstrap_interval_need_not_contain_point(self):
        # Line endpoints remain valid even when the point lies outside a percentile interval.
        d={'decisions':[{'comparison':'test','ci_low':-.2,'ci_high':-.1,'delta_rmse':-.3}]}
        self.assertTrue(visuals.intervals_figure(d).to_json())
    def test_other_result_figures(self):
        d={'pooled_rmse':{'control':1.,'arrival':.9},
           'fits':[{'fold':1,'arm':'control','rmse':1.},{'fold':1,'arm':'arrival','rmse':.9}],
           'slices':[{'fold':1,'arm':'control','slice':'first_second','rmse':.5,'rows':5}]}
        for f in [visuals.pooled_figure,visuals.folds_figure,visuals.slice_figure]:self.assertTrue(f(d).to_json())

    def test_show_save_uses_jupyterlab_renderer(self):
        with tempfile.TemporaryDirectory() as tmp, patch('plotly.graph_objects.Figure.show') as show:
            fig=visuals.origins_figure({'offsets':{'0':{'rows':20,'prethrow_rows':0}}})
            path=visuals.show_save(fig,Path(tmp)/'figure.html')
            self.assertTrue(path.is_file())
            self.assertTrue((Path(tmp)/'plotly.min.js').is_file())
            show.assert_called_once_with(renderer='plotly_mimetype')
