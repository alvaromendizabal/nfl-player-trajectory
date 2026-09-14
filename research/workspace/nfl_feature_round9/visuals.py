"""Plotly views: historical scores and explicitly non-scoring signal diagnostics."""
from pathlib import Path
import json
import numpy as np
import plotly.graph_objects as go


def read(folder,name):
    p=Path(folder)/name
    if not p.is_file():raise FileNotFoundError(f'Complete the preceding stage first: {name}')
    return json.loads(p.read_text())


def layout(fig,title,y=''):
    fig.update_layout(title=title,yaxis_title=y,height=490,margin=dict(l=85,r=25,t=90,b=120),
                      legend=dict(orientation='h',y=1.06),font=dict(size=13))
    return fig


def round8_scores(kit):
    s=read(Path(kit)/'evidence','round8_summary.json');names=['preserved_tree','terminal','history']
    f=go.Figure(go.Bar(x=names,y=[s['metrics'][n] for n in names],text=[f"{s['metrics'][n]:.6f}" for n in names],textposition='outside'))
    return layout(f,'Round 8 · existing matched evaluation (5,886 rows / 14 games)','Coordinate RMSE, yards · lower is better')


def round8_objective_gap(kit):
    e=Path(kit)/'evidence';a=read(e,'round8_models_terminal_complete.json');b=read(e,'round8_models_history_complete.json')
    delta=np.array(b['training_epoch_objectives'])-np.array(a['training_epoch_objectives'])
    f=go.Figure(go.Scatter(x=list(range(1,len(delta)+1)),y=delta,mode='lines+markers'))
    f.update_xaxes(title='Completed epoch')
    return layout(f,'Round 8 · history minus terminal training objective','Objective difference · not validation RMSE')


def round8_horizons(kit):
    s=read(Path(kit)/'evidence','round8_summary.json');r=[r for r in s['slices'] if r['arm']=='history' and r['slice'] in ('first_second','after_first_second')]
    f=go.Figure()
    for key,name in [('rows','Fraction of rows'),('sse','Fraction of squared error')]:
        total=sum(x[key] for x in r);f.add_bar(x=[x['slice'] for x in r],y=[x[key]/total for x in r],name=name)
    f.update_layout(barmode='group');f.update_yaxes(tickformat='.0%')
    return layout(f,'Round 8 · horizon concentration (history arm)','Share · descriptive only')


def feature_support(out):
    s=read(out,'feature_smoke.json');f=go.Figure(go.Bar(x=s['names'],y=s['valid_counts']))
    return layout(f,'New goal-frame channels · training-only measurement support','Valid directed pair/frame observations')


def feature_range(out):
    s=read(out,'feature_smoke.json');f=go.Figure()
    f.add_bar(x=s['names'],y=s['rms_scaled_values'],name='RMS magnitude')
    f.add_bar(x=s['names'],y=s['max_abs_scaled_values'],name='Maximum absolute magnitude')
    return layout(f,'Goal-frame representation · fixed physical scaling','Scaled value · not predictive importance')


def pair_information(out):
    s=read(out,'signal_summary.json');f=go.Figure(go.Bar(x=s['pair_channels'],y=s['input_rms_scaled_difference']))
    return layout(f,'Actual history versus repeated terminal inputs · training only','RMS input difference in original scaled units')


def prediction_changes(out):
    s=read(out,'signal_summary.json');f=go.Figure()
    for arm,r in s['arms'].items():
        f.add_bar(x=[x['variant'] for x in r['prediction_changes']],y=[x['prediction_change_rms_yards'] for x in r['prediction_changes']],name=arm)
    return layout(f,'Frozen-weight stress tests · not RMSE or feature-value evidence','RMS change in predictions, yards')


def channel_gradients(out):
    s=read(out,'signal_summary.json');f=go.Figure()
    for arm,r in s['arms'].items():
        f.add_bar(x=[x['name'] for x in r['channels']],y=[x['signed_projection_gradient_rms'] for x in r['channels']],name=arm)
    return layout(f,'Local pair-input sensitivity · two fixed signed projections','Gradient RMS · not causal importance')


def parameter_movement(out):
    s=read(out,'signal_summary.json');f=go.Figure()
    for arm,r in s['module_weight_changes'].items():
        f.add_bar(x=[x['group'] for x in r],y=[x['relative_delta_l2'] for x in r],name=arm)
    return layout(f,'Saved weights versus their original seeded initialization','Relative L2 change · not feature importance')


def save(fig,out,name):
    folder=Path(out)/'figures';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(name+'.html')
    if path.is_symlink():raise ValueError('Refusing symlink figure')
    fig.write_html(path,include_plotlyjs=True,full_html=True)
    return fig
