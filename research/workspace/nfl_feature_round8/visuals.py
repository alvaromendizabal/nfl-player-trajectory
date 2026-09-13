"""Nine Plotly views. Historical aggregates and new private results are separate."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from parent_support import read_npz


def load(root,name):return json.loads((Path(root)/name).read_text())

def style(fig,title,y=None):
    fig.update_layout(title=title,height=460,font={'size':14},margin={'l':80,'r':30,'t':90,'b':90},legend={'orientation':'h','y':1.12})
    if y:fig.update_yaxes(title=y)
    return fig

def save(fig,out,name):
    d=Path(out)/'figures';d.mkdir(parents=True,exist_ok=True)
    path=d/(name+'.html')
    if path.is_symlink():raise ValueError('Refusing figure symlink')
    fig.write_html(path,include_plotlyjs=True,full_html=True);return fig

def round7_folds(kit):
    s=load(kit,'evidence/round7_summary.json');f=go.Figure()
    for a in ['control','support_only','role_response','player_response']:
        f.add_bar(name=a,x=[str(r['fold']) for r in s['fold_metrics']],y=[r['metrics'][a] for r in s['fold_metrics']])
    return style(f,'Actual Round 7 · later-fold coordinate RMSE','RMSE, yards (lower is better)')

def round7_pooled(kit):
    d=load(kit,'evidence/round7_summary.json')['pooled_metrics']
    f=go.Figure(go.Bar(x=list(d),y=list(d.values()),text=[f'{x:.6f}' for x in d.values()],textposition='outside'))
    return style(f,'Actual Round 7 · pooled results, not Kaggle scores','Coordinate RMSE, yards')

def history_support(kit):
    s=load(kit,'evidence/round7_preparation.json');f=go.Figure()
    for split in ['training','evaluation']:
        f.add_bar(name=split,x=[str(r['fold']) for r in s['folds']],y=[r[split+'_support'][2]['warm_fraction'] for r in s['folds']])
    f.update_yaxes(range=[0,1],tickformat='.0%')
    return style(f,'Actual Round 7 · player-specific history availability','Fraction of rows with earlier player history')

def channel_support(out):
    s=load(out,'smoke.json');f=go.Figure(go.Bar(x=s['pair_channel_names'],y=s['training_pair_support']))
    f.update_yaxes(range=[0,1],tickformat='.0%')
    return style(f,'New training-only smoke · pair-channel measurement support','Valid fraction of jointly observed pair-frames')

def pair_trace(out):
    paths=sorted((Path(out)/'plays').glob('*.npz'))
    z=None
    for p in paths:
        d=read_npz(p)
        if bool(d['train']) and len(d['ids'])>1:z=d;break
    if z is None:raise ValueError('Run the training smoke first')
    from sequence_features import terminal_view
    xy=z['pair'][0,1,:,4]*20;mask=z['pair_valid'][0,1,:,4]
    term=terminal_view(z['pair'],z['pair_valid'])[0,1,:,4]*20
    x=np.arange(-19,1)/10;f=go.Figure()
    for name,v in [('Observed distance history',xy),('Matched terminal control',term)]:
        f.add_scatter(x=x.tolist(),y=[float(a) if b else None for a,b in zip(v,mask)],mode='lines+markers',name=name,connectgaps=False)
    f.update_xaxes(title='Seconds before the forecast origin')
    return style(f,'Private training example · pair selected by canonical order, not errors','Separation, yards')

def learning(out):
    f=go.Figure()
    for a in ['terminal','history']:
        r=load(out,f'models/{a}/complete.json');v=r['training_epoch_objectives']
        f.add_scatter(x=list(range(1,len(v)+1)),y=v,name=a,mode='lines')
    f.update_xaxes(title='Epoch (fixed 24, no validation selection)')
    return style(f,'New matched encoders · training objective during each epoch','Moving-parameter squared-error objective, not held-out RMSE')

def metrics(out):
    d=load(out,'summary.json')['metrics'];f=go.Figure(go.Bar(x=list(d),y=list(d.values()),text=[f'{x:.6f}' for x in d.values()],textposition='outside'))
    return style(f,'New first comparison · only history versus terminal isolates the feature','Coordinate RMSE, yards')

def interval(out):
    c=load(out,'summary.json')['contrast'];d=c['delta_rmse']
    f=go.Figure(go.Scatter(x=['history − terminal'],y=[d],mode='markers',error_y={'type':'data','symmetric':False,'array':[max(0,c['high95']-d)],'arrayminus':[max(0,d-c['low95'])]}))
    f.add_hline(y=0,line_dash='dash')
    return style(f,'Exploratory paired-game interval · reused games, one planned contrast','RMSE difference, yards (negative favors history)')

def horizons(out):
    s=load(out,'summary.json');f=go.Figure()
    for arm in ['terminal','history','preserved_tree']:
        rows=[r for r in s['slices'] if r['arm']==arm and r['slice'] in ['first_second','after_first_second']]
        f.add_bar(name=arm,x=[r['slice'] for r in rows],y=[r['rmse'] for r in rows])
    return style(f,'Diagnostic forecast-horizon errors · no rows removed','Coordinate RMSE, yards')
