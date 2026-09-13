import numpy as np
import pandas as pd
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

