from pathlib import Path
import numpy as np
import pandas as pd
from sequence_features import build
from origin_features import build_view,query_features

def raw_play(game=2023090700,play=1,n=4,t=20):
    rows=[]
    for i in range(n):
        for f in range(1,t+1):
            s=3+i*.2+f*.015;direction=75+i*4
            rows.append(dict(game_id=game,play_id=play,nfl_id=100+i,frame_id=f,
                x=30+i*1.8+f*.3+0.003*f*f*(1+i%2),y=20+i*1.3+.04*f*(i-1),
                play_direction='right',player_role=(['Targeted Receiver','Defensive Coverage','Passer'][i] if i<3 else ('Defensive Coverage' if i%2 else 'Other Route Runner')),
                player_side='Defense' if i%4==1 else 'Offense',player_to_predict=i<2,num_frames_output=16,
                ball_land_x=48.0,ball_land_y=22.0,s=s,dir=direction,o=direction+12))
    return pd.DataFrame(rows)

def sample(raw=None):
    raw=raw_play() if raw is None else raw;f=build(raw,origin=int(raw.frame_id.max()))
    q=np.repeat([100,101],6);times=np.tile(np.array([.1,.3,.5,.8,1.2,1.6]),2)
    v=build_view(raw,origin=int(raw.frame_id.max()));x,_=query_features(v,q,times)
    g=int(raw.game_id.iloc[0]);p=int(raw.play_id.iloc[0]);keys=np.c_[np.repeat(g,len(q)),np.repeat(p,len(q)),q,np.rint(times*10).astype(int)]
    y=np.c_[.2*times*times*(1+(q%2)),np.sin(times)*.12]
    f.update(query=np.array([np.where(f['ids']==i)[0][0] for i in q]),base=x[:,:72],y=y,keys=keys,role_query=q%2,train=np.array(True))
    return f
