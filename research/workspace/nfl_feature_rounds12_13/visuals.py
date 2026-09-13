"""Plotly evidence views: historical aggregates versus new private-run results."""
from pathlib import Path
import json
import numpy as np
import plotly.graph_objects as go

LABELS={'mask':'Availability control','core':'Core six','full':'Full twelve','preserved_tree':'Preserved tree'}

def load(p):return json.loads(Path(p).read_text())
def style(fig,title,y=None):
    fig.update_layout(title=title,height=510,font={'size':14},margin={'l':80,'r':35,'t':95,'b':110},
                      legend={'orientation':'h','y':1.12})
    if y:fig.update_yaxes(title=y)
    return fig

def history_fit(kit):
    d=load(Path(kit)/'evidence/round11_training_audit.json');fig=go.Figure()
    for name,label in [('cartesian','Cartesian encoder'),('goal','Goal encoder')]:
        fig.add_trace(go.Bar(name=label,x=['Training (in-sample)','Historical evaluation'],
            y=[d['metrics'][name]['rmse'],d['historical_evaluation_metrics'][name]]))
    return style(fig,'Round 11 audit • final fit vs historical evaluation','Coordinate RMSE (yards)')

def coverage(kit):
    d=load(Path(kit)/'evidence/round11_coverage_summary.json');used=d['selected_training_plays'];total=d['eligible_observed_plays']
    f=go.Figure(go.Bar(x=['Selected training plays','Other observed plays'],y=[used,total-used],text=[used,total-used],textposition='auto'))
    return style(f,'Same 64 training games • unused-play labels are not verified','Observed play count')

def core_support(kit,r):
    d=load(Path(kit)/'evidence/round11_feature_summary.json')['channel_statistics'];d=d[:6] if r==12 else d[6:]
    f=go.Figure(go.Bar(x=[x['name'] for x in d],y=[100*x['support_fraction'] for x in d]))
    f.update_yaxes(range=[0,100]);return style(f,f'Round {r} • six channels already audited in Round 11','Valid slots (%) including padding')

def support(out,r):
    d=load(Path(out)/'preparation.json')['statistics']
    f=go.Figure(go.Bar(x=[x['name'] for x in d],y=[100*x['support'] for x in d],
        customdata=[[x['valid'],x['slots']] for x in d],hovertemplate='%{x}<br>Support %{y:.2f}%<br>%{customdata[0]} / %{customdata[1]}<extra></extra>'))
    f.update_yaxes(range=[0,100]);return style(f,f'Round {r} • current training-only feature support','Valid slots (%) including padding')

def ranges(out,r):
    from families import SCALES
    d=load(Path(out)/'preparation.json')['statistics'];f=go.Figure()
    for k,label in [('p01','1st percentile'),('rms','RMS'),('p99','99th percentile')]:
        f.add_trace(go.Scatter(name=label,x=[x['name'] for x in d],y=[None if x[k] is None else x[k]/SCALES[r][j] for j,x in enumerate(d)],mode='lines+markers'))
    return style(f,f'Round {r} • feature magnitude after fixed physical scaling','Scaled feature value (training only)')

def learning(out,r):
    f=go.Figure()
    for arm in ('mask','core','full'):
        d=load(Path(out)/f'models/{arm}/complete.json')['training_epoch_objectives']
        f.add_trace(go.Scatter(name=LABELS[arm],x=list(range(1,len(d)+1)),y=d,mode='lines+markers'))
    f.update_xaxes(title='Epoch');return style(f,f'Round {r} • fixed-exposure training; not validation selection','Training objective')

def metrics(out,r):
    d=load(Path(out)/'summary.json');f=go.Figure()
    for name,key in [('Training fit','training_metrics'),('Evaluation','metrics')]:
        f.add_trace(go.Bar(name=name,x=[LABELS[a] for a in d[key]],y=[d[key][a]['rmse'] for a in d[key]]))
    return style(f,f'Round {r} • matched rows; internal result, not Kaggle','Coordinate RMSE (yards)')

def intervals(out,r):
    d=load(Path(out)/'summary.json')['contrasts'];x=list(d);mid=np.array([d[k]['delta_rmse'] for k in x]);lo=np.array([d[k]['low_adjusted'] for k in x]);hi=np.array([d[k]['high_adjusted'] for k in x])
    # Bootstrap percentile intervals need not contain the point estimate. Draw true endpoints.
    f=go.Figure()
    for j,k in enumerate(x):f.add_trace(go.Scatter(x=[j,j],y=[lo[j],hi[j]],mode='lines',showlegend=False,hovertext=k))
    f.add_trace(go.Scatter(x=list(range(len(x))),y=mid,mode='markers',name='RMSE difference'))
    f.update_xaxes(tickvals=list(range(len(x))),ticktext=x);f.add_hline(y=0,line_dash='dash')
    return style(f,f'Round {r} • six-comparison adjusted intervals; reused games','Treatment − control RMSE (yards)')

def horizons(out,r):
    d=load(Path(out)/'summary.json')['slices'];f=go.Figure()
    for arm in ('mask','core','full','preserved_tree'):
        q=[x for x in d if x['arm']==arm and x['slice'] in ('first_second','after_first_second')]
        f.add_trace(go.Bar(name=LABELS[arm],x=[x['slice'] for x in q],y=[x['rmse'] for x in q],customdata=[x['rows'] for x in q],hovertemplate='%{x}<br>RMSE %{y:.5f}<br>Rows %{customdata}<extra></extra>'))
    return style(f,f'Round {r} • horizon slices are diagnostic, not selection rules','Coordinate RMSE (yards)')

def save(fig,out,name):
    folder=Path(out)/'plots';folder.mkdir(parents=True,exist_ok=True)
    fig.write_html(folder/(name+'.html'),include_plotlyjs=True,full_html=True)
    return fig
