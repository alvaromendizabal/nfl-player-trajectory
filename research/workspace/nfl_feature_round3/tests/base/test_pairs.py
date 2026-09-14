import unittest
import numpy as np
import pandas as pd
from pair_features import pair_views,NAMES,ODD
from origin_features import WIDTH,make_example,query_features,build_view


def fixture(frames=20):
    rows=[]
    for player,side,role,y,speed in [(1,'Offense','Targeted Receiver',20,3),(2,'Defense','Defensive Coverage',22,2),(3,'Defense','Defensive Coverage',26,2),(4,'Offense','Other Route Runner',28,3)]:
        for f in range(1,frames+1):
            rows.append(dict(game_id=2023091001,play_id=100,nfl_id=player,frame_id=f,x=30+f*speed/10,y=y,
                             s=speed,dir=90.,o=90.,a=0.,play_direction='right',player_role=role,player_side=side,
                             player_to_predict=player in (1,2),num_frames_output=15,ball_land_x=45.,ball_land_y=20.))
    raw=pd.DataFrame(rows)
    q=np.array([[2023091001,100,p,f] for p in (1,2) for f in range(1,16)],np.int64)
    return raw,q

class PairTests(unittest.TestCase):
    def setUp(self):self.raw,self.q=fixture()
    def runpair(self,raw=None,q=None):return pair_views(self.raw if raw is None else raw,self.q if q is None else q,cutoff=20)
    def test_shape(self):self.assertEqual(self.runpair()['history'].shape,(30,324))
    def test_names_unique(self):self.assertEqual(len(set(NAMES)),len(NAMES))
    def test_finite(self):self.assertTrue(np.isfinite(self.runpair()['history']).all())
    def test_terminal_lag0_equals_history(self):
        z=self.runpair(); ix=[i for i,n in enumerate(NAMES) if '__lag00__' in n];np.testing.assert_array_equal(z['history'][:,ix],z['terminal'][:,ix])
    def test_identical_masks(self):
        z=self.runpair();ix=[i for i,n in enumerate(NAMES) if '__valid__' in n];np.testing.assert_array_equal(z['history'][:,ix],z['terminal'][:,ix])
    def test_history_nontrivial(self):
        z=self.runpair();self.assertGreater(np.abs(z['history']-z['terminal']).max(),0)
    def test_immutable(self):
        old=self.raw.copy(deep=True);q=self.q.copy();self.runpair();pd.testing.assert_frame_equal(old,self.raw);np.testing.assert_array_equal(q,self.q)
    def test_row_permutation(self):np.testing.assert_array_equal(self.runpair()['history'],self.runpair(self.raw.sample(frac=1,random_state=8))['history'])
    def test_query_permutation(self):
        z=self.runpair();np.testing.assert_array_equal(z['history'][::-1],self.runpair(q=self.q[::-1])['history'])
    def test_translation(self):
        raw=self.raw.copy();raw.x+=7;raw.y+=3;raw.ball_land_x+=7;raw.ball_land_y+=3
        np.testing.assert_allclose(self.runpair()['history'],self.runpair(raw)['history'],atol=1e-12)
    def test_lateral_reflection(self):
        raw=self.raw.copy();raw.y=WIDTH-raw.y;raw.ball_land_y=WIDTH-raw.ball_land_y;raw['dir']=(180-raw['dir'])%360;raw.o=(180-raw.o)%360
        want=self.runpair()['history'].copy();want[:,ODD]*=-1
        np.testing.assert_allclose(want,self.runpair(raw)['history'],atol=1e-12)
    def test_left_canonical_parity(self):
        raw=self.raw.copy();raw.x=120-raw.x;raw.y=WIDTH-raw.y;raw.ball_land_x=120-raw.ball_land_x;raw.ball_land_y=WIDTH-raw.ball_land_y
        raw['dir']=(raw['dir']+180)%360;raw.o=(raw.o+180)%360;raw.play_direction='left'
        np.testing.assert_allclose(self.runpair()['history'],self.runpair(raw)['history'],atol=1e-12)
    def test_player_ids_are_keys_only(self):
        raw=self.raw.copy();raw.nfl_id+=100;q=self.q.copy();q[:,2]+=100
        np.testing.assert_array_equal(self.runpair()['history'],self.runpair(raw,q)['history'])
    def test_future_rejected(self):
        raw=self.raw.copy();raw.loc[raw.index[-1],'frame_id']=21
        with self.assertRaises(ValueError):self.runpair(raw)
    def test_future_columns_ignored(self):
        raw=self.raw.copy();raw['future_x']=999999;raw['truth']=np.nan
        np.testing.assert_array_equal(self.runpair()['history'],self.runpair(raw)['history'])
    def test_gap_mask(self):
        raw=self.raw.loc[~((self.raw.nfl_id==1)&(self.raw.frame_id==15))]
        z=self.runpair(raw);ix=NAMES.index('opponent_1__lag05__valid__dx');self.assertEqual(z['history'][0,ix],0)
    def test_stationary_alignment_invalid(self):
        raw=self.raw.copy();raw.s=0;z=self.runpair(raw);i=NAMES.index('opponent_1__lag00__valid__alignment');self.assertEqual(z['history'][0,i],0)
    def test_self_receiver_excluded(self):
        z=self.runpair();ix=[i for i,n in enumerate(NAMES) if n.startswith('targeted_receiver__') and '__valid__' in n];self.assertEqual(z['history'][:15,ix].sum(),0)
    def test_missing_telemetry_fallback(self):self.assertTrue(np.isfinite(self.runpair(self.raw.drop(columns=['s','dir']))['history']).all())
    def test_query_horizon_reject(self):
        q=self.q.copy();q[-1,3]=16
        with self.assertRaises(ValueError):self.runpair(q=q)
    def test_duplicate_query_reject(self):
        with self.assertRaises(ValueError):self.runpair(q=np.r_[self.q,self.q[:1]])
    def test_duplicate_raw_reject(self):
        with self.assertRaises(ValueError):self.runpair(pd.concat([self.raw,self.raw[:1]]))
    def test_wrong_play_reject(self):
        q=self.q.copy();q[:,1]=101
        with self.assertRaises(ValueError):self.runpair(q=q)
    def test_invalid_coordinates(self):
        raw=self.raw.copy();raw.loc[0,'x']=np.nan
        with self.assertRaises(ValueError):self.runpair(raw)
    def test_known_units(self):
        z=self.runpair();i=NAMES.index('opponent_1__lag00__value__dy');self.assertAlmostEqual(z['history'][0,i],.1)
    def test_target_changes_not_features(self):
        y=pd.DataFrame(self.q,columns=['game_id','play_id','nfl_id','frame_id']);y['x']=50.;y['y']=22.
        a=make_example(self.raw,y,0);y.x+=100;b=make_example(self.raw,y,0)
        np.testing.assert_array_equal(a['X'],b['X']);self.assertFalse(np.array_equal(a['y'],b['y']))
if __name__=='__main__':unittest.main()
