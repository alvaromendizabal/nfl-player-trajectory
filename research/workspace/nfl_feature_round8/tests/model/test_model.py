import sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]));sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from fixtures import sample,raw_play
from sequence_model import setup,Forecast,collate,normalization,step,predict,CONFIG,ordered_batches
from neural_study import weight_hash,save_checkpoint,load_checkpoint,compare

class Model(unittest.TestCase):
    def setUp(self):
        setup();self.samples=[sample(),sample(raw_play(play=2,n=6))];self.norm=normalization(self.samples)
    def test_finite_output(self):self.assertTrue(np.isfinite(predict(Forecast(),self.samples,'history',self.norm)).all())
    def test_same_initialization(self):
        setup();a=Forecast();setup();b=Forecast();self.assertEqual(weight_hash(a),weight_hash(b))
    def test_matched_parameter_count(self):
        a=Forecast();b=Forecast();self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
    def test_same_masks_control_and_treatment(self):
        a=collate(self.samples,'history',self.norm);b=collate(self.samples,'terminal',self.norm)
        for k in a:
            if k!='pair':self.assertTrue(torch.equal(a[k],b[k]))
        self.assertFalse(torch.equal(a['pair'],b['pair']))
    def test_targets_ignored_by_collator(self):
        a=collate(self.samples,'history',self.norm);s=[dict(x,y=np.full_like(x['y'],np.nan)) for x in self.samples]
        b=collate(s,'history',self.norm)
        for k in a:self.assertTrue(torch.equal(a[k],b[k]))
    def test_padding_missing_players_finite(self):
        s=sample(raw_play(n=2));m=Forecast();self.assertTrue(np.isfinite(predict(m,[s,self.samples[1]],'terminal',self.norm)).all())
    def test_single_player_mask_all_peers(self):
        s=sample(raw_play());s={k:v for k,v in s.items()}
        # No peer observations, but valid scored players, is allowed.
        s['pair_valid']=np.zeros_like(s['pair_valid']);s['pair']=np.zeros_like(s['pair'])
        self.assertTrue(np.isfinite(predict(Forecast(),[s],'history',self.norm)).all())
    def test_random_order_reproducible(self):
        for a,b in zip(ordered_batches(37,3),ordered_batches(37,3)):np.testing.assert_array_equal(a,b)
    def test_order_covers_all(self):np.testing.assert_array_equal(np.sort(np.concatenate(ordered_batches(37,3))),np.arange(37))
    def test_nonfinite_target_stops(self):
        m=Forecast();o=torch.optim.AdamW(m.parameters());b=collate(self.samples,'history',self.norm)
        with self.assertRaises(ValueError):step(m,o,b,torch.full((len(b['base']),2),float('nan')),10)
    def test_checkpoint_resume_exact(self):
        setup();m=Forecast();o=torch.optim.AdamW(m.parameters(),lr=.001,foreach=False);b=collate(self.samples,'history',self.norm)
        y=torch.tensor(np.concatenate([p['y'] for p in self.samples]),dtype=torch.float32);initial=weight_hash(m)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            for i in range(2):step(m,o,b,y,2*len(y))
            save_checkpoint(folder,m,o,2,'s',[],initial)
            for i in range(2):step(m,o,b,y,2*len(y))
            expected=weight_hash(m)
            n=Forecast();oo=torch.optim.AdamW(n.parameters(),lr=.001,foreach=False);load_checkpoint(folder,n,oo,'s')
            for i in range(2):step(n,oo,b,y,2*len(y))
            self.assertEqual(expected,weight_hash(n));np.testing.assert_array_equal(predict(m,self.samples,'history',self.norm),predict(n,self.samples,'history',self.norm))
    def test_checkpoint_hash_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);m=Forecast();o=torch.optim.AdamW(m.parameters());save_checkpoint(p,m,o,0,'s',[],weight_hash(m))
            f=next(p.glob('*.pt'));f.write_bytes(f.read_bytes()+b'bad')
            with self.assertRaises(ValueError):load_checkpoint(p,m,o,'s')
    def test_checkpoint_signature_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);m=Forecast();o=torch.optim.AdamW(m.parameters());save_checkpoint(p,m,o,0,'s',[],weight_hash(m))
            with self.assertRaises(ValueError):load_checkpoint(p,m,o,'other')
    def test_bootstrap_positive_gain(self):
        y=np.ones((100,2));r=compare(y,np.zeros_like(y),y*.5,np.repeat([1,2,3,4],25))
        self.assertTrue(r['feature_screen_gate']);self.assertLess(r['high95'],0)
    def test_bootstrap_no_improvement(self):
        y=np.ones((100,2));r=compare(y,np.zeros_like(y),np.zeros_like(y),np.repeat([1,2,3,4],25));self.assertFalse(r['feature_screen_gate'])
    def test_mask_poison_ignored_in_model(self):
        m=Forecast();torch.nn.init.normal_(m.decoder[-1].weight,std=.1)
        b=collate(self.samples,'history',self.norm);expected=m(**b).detach()
        b['node'][~b['node_valid']]=float('nan');b['pair'][~b['pair_valid']]=float('nan')
        torch.testing.assert_close(m(**b).detach(),expected,rtol=0,atol=0)
    def test_node_permutation_equivariance(self):
        m=Forecast();torch.nn.init.normal_(m.decoder[-1].weight,std=.1)
        p=self.samples[0];perm=np.array([2,0,3,1]);inv=np.argsort(perm);q=dict(p)
        for k in ['ids','node','node_valid','role','side','node_age']:q[k]=p[k][perm]
        for k in ['pair','pair_valid','pair_age']:q[k]=p[k][perm][:,perm]
        q['query']=inv[p['query']]
        np.testing.assert_allclose(predict(m,[p],'history',self.norm),predict(m,[q],'history',self.norm),atol=1e-6)
    def test_query_permutation(self):
        m=Forecast();torch.nn.init.normal_(m.decoder[-1].weight,std=.1)
        p=self.samples[0];ix=np.arange(len(p['query']))[::-1];q=dict(p)
        q['base']=p['base'][ix];q['query']=p['query'][ix]
        np.testing.assert_allclose(predict(m,[q],'history',self.norm),predict(m,[p],'history',self.norm)[ix],atol=1e-6)
    def test_training_decreases_fixture_objective(self):
        m=Forecast();o=torch.optim.AdamW(m.parameters(),lr=.005,foreach=False);b=collate(self.samples,'history',self.norm)
        y=torch.tensor(np.concatenate([p['y'] for p in self.samples]),dtype=torch.float32)
        before=float((m(**b)-y).square().mean().detach())
        for i in range(12):step(m,o,b,y,2*len(y))
        after=float((m(**b)-y).square().mean().detach());self.assertLess(after,before)
    def test_translation_feature_inputs_not_model_claim(self):
        self.assertEqual(predict(Forecast(),self.samples,'terminal',self.norm).shape,(24,2))
if __name__=='__main__':unittest.main()
