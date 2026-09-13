"""Nine interactive aggregate figures. No synthetic fallback for missing evidence."""
from pathlib import Path
import json
import math
import numpy as np
import plotly.graph_objects as go

LABELS={'control':'Control','direct_state':'Direct observed state','goal_geometry':'State + goal geometry'}


def load(path):
    p=Path(path)
    if not p.is_file():raise FileNotFoundError(f'Complete the preceding notebook stage: {p.name}')
    return json.loads(p.read_text())


def finish(fig,title):
    fig.update_layout(title=title,height=460,font={'size':14},margin={'l':85,'r':35,'t':90,'b':100},
                      legend={'orientation':'h','y':1.06},hovermode='closest')
    return fig


def save(fig,out,name):
    out=Path(out)/'figures';out.mkdir(parents=True,exist_ok=True)
    if Path(name).name!=name:raise ValueError('Unsafe figure name')
    path=out/(name+'.html')
    if path.is_symlink():raise ValueError('Symlink figure path')
    fig.write_html(path,include_plotlyjs=True,full_html=True)
    return fig


def discovery_scores(kit):
    r=load(Path(kit)/'evidence/round4_summary.json');m=r['pooled_rmse']
    fig=go.Figure(go.Bar(x=[LABELS[a] for a in m],y=list(m.values()),text=[f'{x:.6f}' for x in m.values()],textposition='outside'))
    fig.update_yaxes(title='Coordinate RMSE · yards',rangemode='tozero')
    return finish(fig,'Round 4 · actual discovery-fold results (not Kaggle)')


def discovery_intervals(kit):
    r=load(Path(kit)/'evidence/round4_summary.json');cs=r['contrasts']
    x=[(v['adjusted_low']+v['adjusted_high'])/2 for v in cs]
    labels=[LABELS[v['treatment']]+' − '+LABELS[v['control']] for v in cs]
    fig=go.Figure(go.Scatter(x=x,y=labels,mode='markers',marker={'symbol':'line-ns','size':1},
          error_x={'type':'data','array':[(v['adjusted_high']-v['adjusted_low'])/2 for v in cs]},name='Adjusted interval'))
    fig.add_trace(go.Scatter(x=[v['delta_rmse'] for v in cs],y=labels,mode='markers',name='Observed difference',
                            text=[f"Upper bound: {v['adjusted_high']:.10f}" for v in cs]))
    fig.add_vline(x=0,line_dash='dash');fig.update_xaxes(title='Treatment − control RMSE · negative is better')
    return finish(fig,'Round 4 · direct-state upper bound is only −0.000010887')


def discovery_horizon(kit):
    rows=load(Path(kit)/'evidence/round4_summary.json')['slices']
    rows=[r for r in rows if r['arm']=='direct_state' and r['slice'] in ('first_second','after_first_second')]
    n=sum(r['rows'] for r in rows);sse=sum(r['sse'] for r in rows)
    x=['First second','After first second'];fig=go.Figure()
    fig.add_bar(x=x,y=[100*r['rows']/n for r in rows],name='Share of forecast rows')
    fig.add_bar(x=x,y=[100*r['sse']/sse for r in rows],name='Share of squared error')
    fig.update_layout(barmode='group');fig.update_yaxes(title='Percent',range=[0,100])
    return finish(fig,'Round 4 · where direct-state error remains')


def fold_populations(out):
    p=load(Path(out)/'preflight.json')['fold_populations'];fig=go.Figure()
    for key,label in [('training_rows','Training'),('evaluation_rows','Evaluation')]:
        fig.add_bar(x=[f"Fold {v['fold']}" for v in p],y=[v[key] for v in p],name=label)
    fig.update_layout(barmode='group');fig.update_yaxes(title='Forecast rows')
    return finish(fig,'Frozen chronological folds · same selected plays, no new split search')


def cache_reuse(out):
    r=load(Path(out)/'preparation.json');x=['Existing Round 4','Previously completed Round 5','New Round 5']
    y=[r['round4_checkpoints_reused'],r['round5_checkpoints_reused'],r['new_checkpoints']]
    fig=go.Figure(go.Bar(x=x,y=y,text=y,textposition='outside'));fig.update_yaxes(title='Per-play state checkpoints')
    return finish(fig,'Preparation · preserve completed feature work')


def temporal_scores(kit,out):
    r=load(Path(out)/'summary.json');old=load(Path(kit)/'evidence/round4_summary.json')
    fold_names=['Fold 1 · discovery']+[f"Fold {f['fold']} · later" for f in r['fold_results']]
    fig=go.Figure()
    for a in ('control','direct_state'):
        fig.add_bar(x=fold_names,y=[old['pooled_rmse'][a]]+[v['metrics'][a] for v in r['fold_results']],name=LABELS[a])
    fig.update_layout(barmode='group');fig.update_yaxes(title='Coordinate RMSE · yards')
    return finish(fig,'Fixed-estimator feature comparison across time')


def temporal_intervals(out):
    r=load(Path(out)/'summary.json');v=[(f"Fold {f['fold']}",f['comparison']) for f in r['fold_results']]
    v += [('Later folds only',r['pools']['later_folds_only']),('All available · descriptive',r['pools']['all_available_folds'])]
    labels=[x[0] for x in v];rr=[x[1] for x in v]
    fig=go.Figure(go.Scatter(x=[(a['adjusted_low']+a['adjusted_high'])/2 for a in rr],y=labels,mode='markers',
           marker={'symbol':'line-ns','size':1},error_x={'type':'data','array':[(a['adjusted_high']-a['adjusted_low'])/2 for a in rr]},name='Adjusted interval'))
    fig.add_trace(go.Scatter(x=[a['delta_rmse'] for a in rr],y=labels,mode='markers',name='Observed difference'))
    fig.add_vline(x=0,line_dash='dash');fig.update_xaxes(title='Direct state − control RMSE · negative is better')
    return finish(fig,'Later-only evidence controls the replication decision')


def later_horizon(out):
    r=load(Path(out)/'summary.json')['pools']['later_folds_only']['slices'];fig=go.Figure()
    for arm in ('control','direct_state'):
        rows=[v for v in r if v['arm']==arm and v['slice'] in ('first_second','after_first_second')]
        fig.add_bar(x=[v['slice'].replace('_',' ') for v in rows],y=[v['rmse'] for v in rows],name=LABELS[arm])
    fig.update_layout(barmode='group');fig.update_yaxes(title='Coordinate RMSE · yards')
    return finish(fig,'Later folds · horizon diagnostics, not post-hoc selection rules')


def influence_ranges(out):
    r=load(Path(out)/'summary.json')['pools'];labels=[];lo=[];hi=[]
    for key,label in [('later_folds_only','Later only'),('all_available_folds','Including discovery')]:
        d=r[key]['influence'];labels.append(label);lo.append(100*d['delete_one_game_min_gain']);hi.append(100*d['delete_one_game_max_gain'])
    fig=go.Figure(go.Scatter(x=[(a+b)/2 for a,b in zip(lo,hi)],y=labels,mode='markers',
        error_x={'type':'data','array':[(b-a)/2 for a,b in zip(lo,hi)]},name='Delete-one-game range',
        text=[f'Gain range: {a:.3f}% to {b:.3f}%' for a,b in zip(lo,hi)]))
    fig.add_vline(x=0,line_dash='dash');fig.update_xaxes(title='Relative RMSE improvement (%) · positive is better')
    return finish(fig,'Game sensitivity · leave-one-game-out gain ranges (not confidence intervals)')
