"""Interactive evidence views. Missing real evidence is an error, never synthetic filler."""
from pathlib import Path
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pair_features import SLOTS,NAMES,LAGS

LABELS={'control':'Motion control','arrival':'Arrival','terminal_pairs':'Arrival + terminal pairs',
        'history_pairs':'Arrival + pair history','arrival_turn':'Arrival + turning',
        'arrival_brake':'Arrival + braking','arrival_orientation':'Arrival + orientation'}

def finish(fig,title,x=None,y=None):
    fig.update_layout(title=title,font=dict(size=14),height=470,width=1000,
                      margin=dict(l=90,r=30,t=80,b=130),legend=dict(orientation='h',y=1.07),
                      xaxis_title=x,yaxis_title=y)
    return fig

def load(path):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'Required result not present: {path.name}. Run the preceding stage; no substitute data is shown.')
    return json.loads(path.read_text())

def previous_results(kit):
    d=load(Path(kit)/'evidence/round2_motion_summary.json')
    fig=go.Figure(go.Bar(x=[LABELS.get(a,a) for a in d['pooled_rmse']],y=list(d['pooled_rmse'].values()),
                        text=[f'{v:.6f}' for v in d['pooled_rmse'].values()],textposition='outside'))
    return finish(fig,'Round 2 actual results — reused 4,685 rows; NOT Kaggle scores',y='Coordinate RMSE (yards)')

def removals(kit):
    d=load(Path(kit)/'evidence/round2_attribution_summary.json')
    rows=[r for r in d['decisions'] if r['direction']=='removal']
    fig=go.Figure(go.Bar(x=[r['family'] for r in rows],y=[r['delta_rmse'] for r in rows]))
    return finish(fig,'Arrival component removal — positive means worse',x='Removed component',y='RMSE change from full arrival (yards)')

def sample_history(out):
    out=Path(out);s=load(out/'selection.json');tr=set(s['folds'][0]['train_games'])
    p=next(p for p in s['plays'] if p['game'] in tr)
    with np.load(out/'features/plays'/f"{p['game']}_{p['play']}.npz",allow_pickle=False) as z:
        h=z['history'][0];t=z['terminal'][0]
    ix=[NAMES.index(f'opponent_1__lag{lag:02d}__value__distance') for lag in LAGS]
    mi=[NAMES.index(f'opponent_1__lag{lag:02d}__valid__distance') for lag in LAGS]
    order=np.argsort(-np.asarray(LAGS));xx=(-np.asarray(LAGS)/10)[order]
    fig=go.Figure()
    for name,values in [('Observed pair history',h),('Repeated terminal control',t)]:
        yy=[float(values[ix[k]]*20) if h[mi[k]] else None for k in order]
        fig.add_trace(go.Scatter(x=xx,y=yy,mode='lines+markers',name=name,connectgaps=False))
    return finish(fig,'First training play — fixed opponent identity, real observation gaps',x='Seconds before prediction origin',y='Separation (yards)')

def sample_balance(out):
    s=load(Path(out)/'selection.json');df=pd.DataFrame(s['plays']);games=sorted(df.game.unique());rows=[]
    for i,g in enumerate(games,1):
        for old in (True,False):rows.append({'game_order':i,'source':'Preserved plays' if old else 'New original-origin plays','plays':int(((df.game==g)&(df.old==old)).sum())})
    return finish(px.bar(pd.DataFrame(rows),x='game_order',y='plays',color='source'),
                  'Expanded sample — original plays retained, additions spread across games',x='Training-side game order',y='Selected plays')

def support(out):
    d=load(Path(out)/'preparation.json')
    return finish(go.Figure(go.Bar(x=SLOTS,y=d['mean_pair_history_support'])),
                  'Observed pair support — first-fold training players only',y='Mean fraction of jointly observed frames')

def first_fold(out):
    d=load(Path(out)/'fold_1/summary.json')
    return finish(go.Figure(go.Bar(x=[LABELS[k] for k in d['pooled_rmse']],y=list(d['pooled_rmse'].values()),
                  text=[f'{v:.6f}' for v in d['pooled_rmse'].values()],textposition='outside')),
                  'Round 3 first fold — compare arms here, not against Round 2 RMSE',y='Coordinate RMSE (yards)')

def contrasts(out):
    d=load(Path(out)/'fold_1/summary.json');fig=go.Figure()
    for r in d['contrasts']:
        label=f"{LABELS[r['treatment']]} vs {LABELS[r['control']]}"
        fig.add_trace(go.Scatter(x=[r['adjusted_low'],r['adjusted_high']],y=[label,label],mode='lines',showlegend=False))
        fig.add_trace(go.Scatter(x=[r['delta_rmse']],y=[label],mode='markers',showlegend=False))
    fig.add_vline(x=0,line_dash='dot')
    return finish(fig,'Paired-game intervals — negative favors the added representation',x='RMSE difference (yards)')

def slices(out,kind='horizon'):
    d=load(Path(out)/'fold_1/summary.json');rows=[]
    for r in d['slices']:
        if (kind=='horizon' and r['slice'].startswith('role_')) or (kind=='role' and not r['slice'].startswith('role_')):continue
        rows.append({**r,'model':LABELS[r['arm']]})
    return finish(px.bar(pd.DataFrame(rows),x='slice',y='rmse',color='model',barmode='group'),
                  ('Horizon error' if kind=='horizon' else 'Role error')+' — descriptive, no post-hoc role gating',y='Coordinate RMSE (yards)')

def save(fig,out,name):
    folder=Path(out)/'figures';folder.mkdir(parents=True,exist_ok=True)
    fig.write_html(folder/(name+'.html'),include_plotlyjs=True,full_html=True)
    return fig
