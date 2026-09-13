"""Plotly reviews of recorded aggregates. Missing results are never synthesized."""
from pathlib import Path
import json
import numpy as np
import plotly.graph_objects as go
from state_features import DIRECT_NAMES

LABELS={'control':'Preserved control','arrival':'Arrival','terminal_pairs':'Terminal relationships',
        'history_pairs':'Relationship histories','direct_state':'Direct observed state',
        'goal_geometry':'Direct state + goal geometry'}
ROLES={'role_0':'Targeted Receiver','role_1':'Defensive Coverage','role_2':'Passer','role_3':'Other Route Runner'}


def read(path):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'Run the prerequisite stage: {path.name}')
    return json.loads(path.read_text())


def style(fig,title,*,height=500,y=None):
    fig.update_layout(title=title,height=height,font={'size':14},margin={'l':80,'r':45,'t':90,'b':110},
                      legend={'orientation':'h','y':1.08},hovermode='closest')
    if y:fig.update_yaxes(title=y)
    return fig


def scores(summary,title):
    metrics=summary['pooled_rmse']
    fig=go.Figure(go.Bar(x=[LABELS.get(k,k) for k in metrics],y=list(metrics.values()),
                         text=[f'{v:.6f}' for v in metrics.values()],textposition='outside'))
    return style(fig,title,y='Coordinate RMSE (yards; lower is better)')


def contrasts(summary,title):
    rows=summary['contrasts']
    labels=[LABELS.get(r['treatment'],r['treatment'])+' − '+LABELS.get(r['control'],r['control']) for r in rows]
    lx=[];ly=[]
    for r,label in zip(rows,labels):
        lx.extend([r['adjusted_low'],r['adjusted_high'],None]);ly.extend([label,label,None])
    fig=go.Figure(go.Scatter(x=lx,y=ly,mode='lines',name='Adjusted interval',line={'width':4},hoverinfo='skip'))
    fig.add_scatter(x=[r['delta_rmse'] for r in rows],y=labels,mode='markers',name='Observed difference',marker={'size':12},
        customdata=[[r['adjusted_low'],r['adjusted_high'],r['relative_gain']*100] for r in rows],
        hovertemplate='Delta: %{x:.6f}<br>Adjusted interval: [%{customdata[0]:.6f}, %{customdata[1]:.6f}]<br>Gain: %{customdata[2]:.3f}%<extra></extra>')
    fig.add_vline(x=0,line_dash='dash')
    style(fig,title,height=510)
    fig.update_layout(margin={'l':385,'r':45,'t':110,'b':90})
    fig.update_xaxes(title='RMSE difference (negative favors the added representation)')
    return fig


def error_concentration(summary):
    rows={r['slice']:r for r in summary['slices'] if r['arm']=='control' and r['slice'] in ('first_second','after_first_second')}
    names=['first_second','after_first_second'];labels=['First second','After first second']
    n=sum(rows[k]['rows'] for k in names);s=sum(rows[k]['sse'] for k in names)
    fig=go.Figure()
    for label,col,den in [('Share of forecast rows','rows',n),('Share of squared error','sse',s)]:
        vals=[100*rows[k][col]/den for k in names]
        fig.add_bar(name=label,x=labels,y=vals,text=[f'{v:.2f}%' for v in vals],textposition='outside')
    fig.update_layout(barmode='group')
    return style(fig,'Round 3 control · Where the error remains',y='Percent')


def support(preparation):
    names=preparation['support_columns'];values=preparation['training_support_fractions']
    fig=go.Figure(go.Bar(x=[n.replace('_',' ') for n in names],y=values,text=[f'{v:.1%}' for v in values],textposition='outside'))
    fig.update_yaxes(range=[0,1.13],tickformat='.0%')
    return style(fig,'Training-only measurement support · one count per scored player/play',height=560,y='Fraction available')


def state_ranges(preparation):
    lo=np.asarray(preparation['state_train_min']);hi=np.asarray(preparation['state_train_max']);mid=(lo+hi)/2
    fig=go.Figure(go.Scatter(x=mid,y=[n.removeprefix('observed__') for n in DIRECT_NAMES],mode='markers',
        error_x={'type':'data','symmetric':False,'array':hi-mid,'arrayminus':mid-lo},
        customdata=np.column_stack([lo,hi]),hovertemplate='%{y}<br>Train min: %{customdata[0]:.4f}<br>Train max: %{customdata[1]:.4f}<extra></extra>'))
    style(fig,'Observed state · training-only feature ranges (legacy fixed units)',height=1250)
    fig.update_layout(margin={'l':280,'r':45,'t':90,'b':75});fig.update_xaxes(title='Legacy fixed-scale feature value; not multiplied by query time')
    return fig


def horizon_errors(summary):
    fig=go.Figure()
    for arm in summary['pooled_rmse']:
        rows={r['slice']:r for r in summary['slices'] if r['arm']==arm}
        names=[n for n in ('first_second','after_first_second') if n in rows]
        fig.add_bar(name=LABELS.get(arm,arm),x=[n.replace('_',' ') for n in names],y=[rows[n]['rmse'] for n in names])
    fig.update_layout(barmode='group')
    return style(fig,'Round 4 · Forecast-horizon errors (descriptive, not a selection gate)',y='Coordinate RMSE (yards)')


def role_errors(summary):
    fig=go.Figure()
    for arm in summary['pooled_rmse']:
        rows={r['slice']:r for r in summary['slices'] if r['arm']==arm and r['slice'] in ROLES}
        fig.add_bar(name=LABELS.get(arm,arm),x=[ROLES[k] for k in rows],y=[r['rmse'] for r in rows.values()],
                    customdata=[r['rows'] for r in rows.values()],hovertemplate='%{x}<br>RMSE: %{y:.6f}<br>Rows: %{customdata}<extra></extra>')
    fig.update_layout(barmode='group')
    return style(fig,'Round 4 · Role errors (no post-hoc role gating)',y='Coordinate RMSE (yards)')


def save(fig,out,name):
    out=Path(out)/'html';out.mkdir(parents=True,exist_ok=True)
    if Path(name).name!=name:raise ValueError('Unsafe figure name')
    fig.write_html(out/(name+'.html'),include_plotlyjs=True,full_html=True)
    return fig
