"""Plotly reviews: historical real aggregates or current local reports only."""
from pathlib import Path
import json
import numpy as np
import plotly.graph_objects as go

LABELS={'control':'Preserved control','pass_axis':'Passing-line state','pass_axis_history':'Passing-line history',
        'support_only':'History support only','role_response':'Role response','player_response':'Player + role response'}

def load(root,name):
    p=Path(root)/name
    if not p.is_file():raise FileNotFoundError(f'{name} is missing; run its preceding stage, not a fabricated preview.')
    return json.loads(p.read_text())

def finish(fig,title,y):
    fig.update_layout(title=title,xaxis_title='',yaxis_title=y,height=470,template='plotly_white',
                      font={'size':14},margin={'l':90,'r':30,'t':90,'b':100},legend={'orientation':'h','y':1.06})
    return fig

def save(fig,out,name):
    p=Path(out)/'figures';p.mkdir(parents=True,exist_ok=True)
    fig.write_html(p/(name+'.html'),include_plotlyjs=True,full_html=True)
    return fig

def previous_metrics(kit):
    r=load(Path(kit)/'evidence/round6','summary.json');fig=go.Figure()
    for arm in ('control','pass_axis','pass_axis_history'):
        fig.add_bar(name=LABELS[arm],x=[f'Later fold {f["fold"]}' for f in r['fold_metrics']],y=[f['metrics'][arm] for f in r['fold_metrics']])
    return finish(fig,'Round 6 • actual uploaded results, not Round 7','Coordinate RMSE (yards)')

def interval_plot(contrasts,title):
    labels=[LABELS[v['treatment']]+' / '+LABELS[v['control']] for v in contrasts]
    fig=go.Figure()
    for label,v in zip(labels,contrasts):
        fig.add_scatter(x=[v['adjusted_low'],v['adjusted_high']],y=[label,label],
                        mode='lines',showlegend=False,hovertemplate='%{x:.6f} yards<extra></extra>')
    fig.add_scatter(x=[v['delta_rmse'] for v in contrasts],y=labels,mode='markers',
                    name='Observed difference',marker={'size':10})
    fig.add_vline(x=0,line_dash='dash')
    finish(fig,title,'')
    fig.update_layout(xaxis_title='Treatment − reference RMSE (yards)',
                      height=max(450,100+75*len(labels)),margin={'l':320,'r':35,'t':90,'b':60})
    return fig

def previous_intervals(kit):
    return interval_plot(load(Path(kit)/'evidence/round6','summary.json')['contrasts'],
                         'Round 6 • all adjusted intervals cross zero')

def previous_error_budget(kit):
    rows=[]
    for f in (2,3):rows+=load(Path(kit)/'evidence/round6',f'fold_{f}/summary.json')['slices']
    a=[r for r in rows if r['arm']=='control' and r['slice'] in ('first_second','after_first_second')]
    n=sum(v['rows'] for v in a);s=sum(v['sse'] for v in a);labels=['first_second','after_first_second'];fig=go.Figure()
    fig.add_bar(name='Forecast rows',x=['First second','After first second'],y=[100*sum(v['rows'] for v in a if v['slice']==k)/n for k in labels])
    fig.add_bar(name='Squared error',x=['First second','After first second'],y=[100*sum(v['sse'] for v in a if v['slice']==k)/s for k in labels])
    return finish(fig,'Round 6 • preserved control error concentration','Share (%)')

def support_coverage(out):
    r=load(out,'preparation.json');fig=go.Figure()
    for partition in ('training','evaluation'):
        x=[];y=[]
        for f in r['folds']:
            for a in f[partition+'_support']:
                x.append(f'Fold {f["fold"]}<br>'+a['level'].replace('_',' '));y.append(100*a['warm_fraction'])
        fig.add_bar(name=partition.title(),x=x,y=y)
    return finish(fig,'Historical support • not feature importance','Rows with earlier matching history (%)')

def support_counts(out):
    r=load(out,'preparation.json');fig=go.Figure()
    for partition in ('training','evaluation'):
        x=[];y=[]
        for f in r['folds']:
            for a in f[partition+'_support']:
                x.append(f'Fold {f["fold"]}<br>'+a['level'].replace('_',' '));y.append(a['median_prior_trajectories'])
        fig.add_bar(name=partition.title(),x=x,y=y)
    return finish(fig,'Median donor support • each trajectory contributes once per phase','Prior player-play-phase contributions')

def current_metrics(out):
    r=load(out,'summary.json');fig=go.Figure()
    for arm in ('control','support_only','role_response','player_response'):
        fig.add_scatter(name=LABELS[arm],x=[f'Fold {f["fold"]}' for f in r['fold_metrics']],y=[f['metrics'][arm] for f in r['fold_metrics']],mode='lines+markers')
    return finish(fig,'Round 7 • matched historical-feature ablation','Coordinate RMSE (yards)')

def current_intervals(out):
    return interval_plot(load(out,'summary.json')['contrasts'],
                         'Reused-game uncertainty • 18 planned comparisons')

def slice_plot(out,labels,title):
    r=load(out,'summary.json');s=[]
    for fold in r['folds']:s+=load(out,f'fold_{fold}/summary.json')['slices']
    fig=go.Figure()
    for arm in ('control','support_only','role_response','player_response'):
        yy=[]
        for label in labels:
            a=[v for v in s if v['arm']==arm and v['slice']==label];n=sum(v['rows'] for v in a)
            yy.append(float(np.sqrt(sum(v['sse'] for v in a)/(2*n))) if n else None)
        fig.add_bar(name=LABELS[arm],x=[k.replace('_',' ').title() for k in labels],y=yy)
    return finish(fig,title,'Coordinate RMSE (yards)')

def horizon_errors(out):return slice_plot(out,['first_second','after_first_second'],'Horizon diagnosis • no row removal or post-hoc model selection')
def cold_start_errors(out):return slice_plot(out,['player_cold','player_warm'],'Player-history availability • diagnostic, not a role-gated ensemble')
