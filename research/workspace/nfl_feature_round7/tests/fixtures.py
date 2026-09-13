from datetime import date,timedelta
import numpy as np
from history_features import Metadata


def meta_fixture(days=6, games_per_day=2, players=3, frames=12):
    keys=[];roles=[];times=[];horizons=[];targets=[]
    for d in range(days):
        stamp=int((date(2023,9,1)+timedelta(days=d*7)).strftime('%Y%m%d'))
        for g in range(games_per_day):
            for p in range(players):
                for f in range(1,frames+1):
                    t=f/10
                    keys.append([stamp*100+g,100+g,p+1,f]);roles.append(p%2)
                    times.append(t);horizons.append(frames/10)
                    targets.append([t*(p+1)+d*.01,-t*(p+2)])
    m=Metadata(np.asarray(keys,np.int64),np.asarray(roles,np.int8),np.asarray(times),np.asarray(horizons))
    return m,np.asarray(targets)
