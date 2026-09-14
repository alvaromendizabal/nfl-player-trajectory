"""Plotly evidence views. No missing result is replaced by simulated NFL data."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder


def read(root,name):
    p=Path(root)/name
    if not p.is_file():raise FileNotFoundError(f'Run the preceding stage first: {p.name}')
    return json.loads(p.read_text())


def polish(fig,title,yaxis):
    return fig.update_layout(title=title,yaxis_title=yaxis,font={'size':14},height=490,
                             margin={'t':85,'b':95,'l':90,'r':35},legend={'orientation':'h','y':1.1})


def save(fig,out,name):
    out=Path(out)/'figures';out.mkdir(parents=True,exist_ok=True)
    # Offline HTML is self-contained; no CDN dependency in private review.
    fig.write_html(out/(name+'.html'),include_plotlyjs=True,full_html=True)
    json.dumps(fig.to_plotly_json(),cls=PlotlyJSONEncoder,allow_nan=False)
    return fig


def prior_inputs(kit):
    s=read(Path(kit)/'evidence','round9_signal_summary.json')
    f=go.Figure(go.Bar(x=s['pair_channels'],y=np.asarray(s['different_fraction'])*100))
    return polish(f,'Round 9 · History inputs genuinely differed','Different valid elements (%)')


def prior_sensitivity(kit):
    s=read(Path(kit)/'evidence','round9_signal_summary.json')
    f=go.Figure()
    labels={'swap_pair_view':'Swap history/terminal','zero_pair_values_keep_masks':'Zero pair values',
            'zero_decoder_context':'Zero relationship context','zero_decoder_node':'Zero own-motion context',
            'zero_decoder_base':'Zero base inputs'}
    for arm in ('terminal','history'):
        rows=[r for r in s['arms'][arm]['prediction_changes'] if r['variant'] in labels]
        f.add_bar(name=arm,x=[labels[r['variant']] for r in rows],y=[r['prediction_change_rms_yards'] for r in rows])
    f.update_yaxes(type='log')
    f.add_annotation(text='Training-input interventions, potentially off-distribution. Not prediction error or causal importance.',
                     xref='paper',yref='paper',x=0,y=-.30,showarrow=False)
    return polish(f,'Round 9 · Small response to pair-history changes','Prediction-change RMS (yards; log scale)')


def prior_weights(kit):
    s=read(Path(kit)/'evidence','round9_signal_summary.json');f=go.Figure()
    for arm,rows in s['module_weight_changes'].items():
        f.add_bar(name=arm,x=[r['group'] for r in rows],y=[r['relative_delta_l2'] for r in rows])
    return polish(f,'Round 9 · Relationship weights were not frozen','Weight change / initial norm')


def support(out):
    s=read(out,'preparation.json');f=go.Figure(go.Bar(x=s['channels'],y=s['training_valid_counts']))
    return polish(f,'Round 10 · Available goal-frame measurements','Training valid elements')


def magnitude(out):
    s=read(out,'preparation.json');f=go.Figure(go.Bar(x=s['channels'],y=s['training_rms_scaled']))
    return polish(f,'Round 10 · Physical scaling of the six candidates','Training RMS in fixed scaled units')


def learning(out):
    f=go.Figure()
    for arm in ('cartesian','goal'):
        s=read(out,f'models/{arm}/complete.json');v=s['training_epoch_objectives']
        f.add_scatter(x=list(range(1,len(v)+1)),y=v,mode='lines+markers',name=arm)
    return polish(f,'Round 10 · Fixed-exposure training, no validation selection','Mean training objective')


def metrics(out):
    s=read(out,'summary.json');m=s['metrics'];f=go.Figure(go.Bar(x=list(m),y=list(m.values()),
        text=[f'{v:.6f}' for v in m.values()],textposition='outside'))
    return polish(f,'Round 10 · Matched evaluation rows','Coordinate RMSE (yards)')


def interval(out):
    s=read(out,'summary.json')['contrast'];d=s['delta_rmse']
    f=go.Figure(go.Scatter(x=[d],y=['Goal − Cartesian'],mode='markers',
        error_x={'type':'data','symmetric':False,'array':[max(0,s['high95']-d)],'arrayminus':[max(0,d-s['low95'])]}))
    # Exact percentile endpoints are also drawn; they need not bracket the estimate.
    f.add_shape(type='line',x0=s['low95'],x1=s['high95'],y0=0,y1=0)
    f.add_vline(x=0,line_dash='dash')
    f.update_layout(title='Round 10 · One planned paired-game contrast',height=340,
                    xaxis_title='Coordinate RMSE difference (yards; lower favors goal)',font={'size':14})
    return f


def horizons(out):
    rows=read(out,'summary.json')['slices'];f=go.Figure()
    for arm in ('cartesian','goal','preserved_tree'):
        rr=[r for r in rows if r['arm']==arm and r['slice'] in ('first_second','after_first_second')]
        f.add_bar(name=arm,x=[r['slice'] for r in rr],y=[r['rmse'] for r in rr])
    return polish(f,'Round 10 · Horizon slices, all rows retained','Coordinate RMSE (yards)')
