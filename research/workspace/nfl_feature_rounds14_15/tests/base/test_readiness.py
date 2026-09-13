import tempfile,unittest,json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from unittest.mock import patch
from readiness import require_review_release
from label_audit import audit_week
from audit_io import safe_file,checkpoint_npz,digest
import numpy as np

class Readiness(unittest.TestCase):
    def test_no_implicit_release(self):
        with tempfile.TemporaryDirectory() as d:
            a=SimpleNamespace(out=Path(d),round_no=14)
            with patch('readiness.reviewed_readiness',return_value={'source_signature':'x','readiness_hash':'y'}):
                with self.assertRaises(ValueError):require_review_release(a)
    def test_release_cannot_ignore_prior_failures(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'review_release.json').write_text(json.dumps({'decision':'bounded_exploratory_study_after_readiness_review'}))
            with patch('readiness.reviewed_readiness',return_value={'source_signature':'x','readiness_hash':'y'}):
                with self.assertRaises(ValueError):require_review_release(SimpleNamespace(out=p,round_no=14))
    def test_release_complete(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r={'decision':'bounded_exploratory_study_after_readiness_review','readiness_hash':'y','source_signature':'x','round':14,'max_scientific_models':3,'acknowledge_prior_gates_failed':True,'acknowledge_reused_evaluation':True,'reason':'Reviewed independently; another controlled experiment is justified.','data_scale_decision':'Data scale is fixed here; larger training data is a separate comparison.'}
            (p/'review_release.json').write_text(json.dumps(r))
            with patch('readiness.reviewed_readiness',return_value={'source_signature':'x','readiness_hash':'y'}):
                self.assertEqual(require_review_release(SimpleNamespace(out=p,round_no=14)),r)
    def test_replay_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.npz';x={'values':np.ones((2,3),np.float32),'valid':np.ones((2,3),bool)}
            h=checkpoint_npz(p,x);checkpoint_npz(p,x);self.assertEqual(digest(p),h)
    def test_replay_rejects_dtype(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.npz';checkpoint_npz(p,{'x':np.ones(2,np.float32)})
            with self.assertRaises(ValueError):checkpoint_npz(p,{'x':np.ones(2,np.float64)})
    def test_paths(self):
        with self.assertRaises(ValueError):safe_file(Path('/tmp'),'../escape')

class Labels(unittest.TestCase):
    def files(self,p):
        # Deliberately shuffle CSV column order to exercise organizer-column handling.
        x=pd.DataFrame({'player_to_predict':[True,True,True],'frame_id':[1,2,1],
                        'nfl_id':[5,5,6],'game_id':[2023091000,2023091000,2023091700],
                        'num_frames_output':[2,2,1],'play_id':[7,7,8]})
        y=pd.DataFrame({'y':[2.,3.,999.],'x':[1.,2.,999.],'frame_id':[1,2,1],
                        'nfl_id':[5,5,6],'play_id':[7,7,8],'game_id':[2023091000,2023091000,2023091700]})
        a=p/'input.csv';b=p/'output.csv';x.to_csv(a,index=False);y.to_csv(b,index=False)
        return a,b,x,y
    def test_complete(self):
        with tempfile.TemporaryDirectory() as d:
            a,b,_,_=self.files(Path(d));r=audit_week(a,b,{2023091000})
            self.assertEqual(r['counts']['eligible_plays'],1);self.assertEqual(r['counts']['expected_rows'],2)
    def test_validation_labels_unused(self):
        with tempfile.TemporaryDirectory() as d:
            a,b,x,y=self.files(Path(d));r=audit_week(a,b,{2023091000})
            y['x']=y['x'].astype(object);y.loc[y.game_id==2023091700,'x']='not a number';y.to_csv(b,index=False)
            self.assertEqual(audit_week(a,b,{2023091000}),r)
    def test_missing_row(self):
        with tempfile.TemporaryDirectory() as d:
            a,b,x,y=self.files(Path(d));y=y.iloc[1:];y.to_csv(b,index=False)
            r=audit_week(a,b,{2023091000});self.assertEqual(r['counts']['missing_rows'],1);self.assertEqual(r['counts']['eligible_plays'],0)
    def test_duplicate_target(self):
        with tempfile.TemporaryDirectory() as d:
            a,b,x,y=self.files(Path(d));pd.concat([y,y.iloc[:1]]).to_csv(b,index=False)
            self.assertEqual(audit_week(a,b,{2023091000})['counts']['duplicate_rows'],1)
    def test_nonfinite_target(self):
        with tempfile.TemporaryDirectory() as d:
            a,b,x,y=self.files(Path(d));y.loc[0,'x']=float('inf');y.to_csv(b,index=False)
            self.assertEqual(audit_week(a,b,{2023091000})['counts']['nonfinite_rows'],1)
    def test_unexpected_horizon(self):
        with tempfile.TemporaryDirectory() as d:
            a,b,x,y=self.files(Path(d));y.loc[1,'frame_id']=3;y.to_csv(b,index=False)
            r=audit_week(a,b,{2023091000});self.assertEqual(r['counts']['unexpected_rows'],1);self.assertEqual(r['counts']['missing_rows'],1)
    def test_duplicate_observation(self):
        with tempfile.TemporaryDirectory() as d:
            a,b,x,y=self.files(Path(d));pd.concat([x,x.iloc[:1]]).to_csv(a,index=False)
            with self.assertRaises(ValueError):audit_week(a,b,{2023091000})
if __name__=='__main__':unittest.main()
