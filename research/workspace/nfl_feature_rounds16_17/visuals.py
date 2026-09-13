"""Plotly figures from supplied evidence or user-created receipts; no fallback scores."""
from pathlib import Path
import json
import plotly.graph_objects as go

def read(path):
    p=Path(path)
    if not p.is_file():raise FileNotFoundError('Required evidence missing: '+str(p))
    return json.loads(p.read_text())

def style(fig,title,x,y):
    return fig.update_layout(title={'text':title,'x':.02},xaxis_title=x,yaxis_title=y,
                             height=470,template='plotly_white',font={'size':14},
                             margin={'l':80,'r':35,'t':95,'b':100},legend={'orientation':'h','y':1.10})

def historical_metrics(kit,prior):
    s=read(Path(kit)/'evidence'/f'round{prior}'/'summary.json')
    names=list(s['metrics']);v=[s['metrics'][n]['rmse'] for n in names]
    fig=go.Figure(go.Bar(x=names,y=v,text=[f'{x:.6f}' for x in v],textposition='outside'))
    return style(fig,f'Uploaded Round {prior} — internal evaluation, not Kaggle','Arm','Coordinate RMSE (yards)')

def historical_roles(kit,prior):
    s=read(Path(kit)/'evidence'/f'round{prior}'/'summary.json');fig=go.Figure()
    for arm in ('mask','core','full','preserved_tree'):
        rows=[r for r in s['slices'] if r['arm']==arm and r['slice'].startswith('role_')]
        fig.add_bar(name=arm,x=[r['slice'] for r in rows],y=[r['rmse'] for r in rows])
    fig.update_layout(barmode='group')
    return style(fig,'Uploaded role errors — do not select role-specific models post hoc','Supplied role code (0 receiver, 1 coverage)','Coordinate RMSE (yards)')

def coverage(label_path):
    s=read(label_path)
    fig=go.Figure(go.Bar(x=['Selected training plays','Additional label-eligible plays','Not label-eligible'],
                        y=[s['selected_training_plays'],s['additional_label_eligible_plays'],
                           s['observed_training_plays']-s['label_eligible_training_plays']]))
    return style(fig,'Same frozen training games — label eligibility is not model performance','Population','Number of plays')

def support(out):
    s=read(Path(out)/'smoke.json');r=s['statistics']
    fig=go.Figure(go.Bar(x=[x['name'] for x in r],y=[100*x['support'] for x in r]))
    fig.update_xaxes(tickangle=-40)
    return style(fig,'32 training plays — observed feature availability','Channel','Valid slots (%) including padded slots')

def magnitude(out):
    s=read(Path(out)/'smoke.json');r=s['statistics'];fig=go.Figure()
    fig.add_bar(name='Root mean square',x=[x['name'] for x in r],y=[x['rms'] for x in r])
    fig.add_scatter(name='Maximum absolute',x=[x['name'] for x in r],y=[x['max_abs'] for x in r],mode='markers',
                    customdata=[x['unit'] for x in r],hovertemplate='%{x}<br>%{y:.5g} %{customdata}<extra></extra>')
    fig.update_xaxes(tickangle=-40)
    return style(fig,'Training-only magnitude diagnostics — units differ across channels','Channel','Physical magnitude (see channel unit)')

def learning(out):
    fig=go.Figure()
    for arm in ('mask','core','full'):
        r=read(Path(out)/'models'/arm/'complete.json');y=r['training_epoch_objectives']
        fig.add_scatter(name=arm,x=list(range(1,len(y)+1)),y=y,mode='lines')
    return style(fig,'Training objective during optimization — not final validation RMSE','Epoch','Mean recorded objective')

def metrics(out):
    s=read(Path(out)/'summary.json');fig=go.Figure()
    for setting,field in [('Training','training_metrics'),('Reused evaluation','metrics')]:
        rows=s[field];fig.add_bar(name=setting,x=list(rows),y=[x['rmse'] for x in rows.values()])
    fig.update_layout(barmode='group')
    return style(fig,'Final checkpoint — separate training and reused evaluation','Arm','Coordinate RMSE (yards)')

def contrasts(out):
    c=read(Path(out)/'summary.json')['contrasts'];names=list(c)
    x=[c[n]['delta_rmse'] for n in names];high=[max(0,c[n]['high_adjusted']-c[n]['delta_rmse']) for n in names]
    low=[max(0,c[n]['delta_rmse']-c[n]['low_adjusted']) for n in names]
    fig=go.Figure(go.Scatter(x=x,y=names,mode='markers',error_x={'type':'data','symmetric':False,'array':high,'arrayminus':low}))
    fig.add_vline(x=0,line_dash='dash')
    return style(fig,'Paired game bootstrap — adjustment for six planned comparisons only','Treatment minus control RMSE (yards; negative is better)','Planned comparison')

def horizons(out):
    s=read(Path(out)/'summary.json');fig=go.Figure()
    for arm in ('mask','core','full','preserved_tree'):
        rows=[r for r in s['slices'] if r['arm']==arm and r['slice'] in ('first_second','after_first_second')]
        fig.add_bar(name=arm,x=[r['slice'] for r in rows],y=[r['rmse'] for r in rows])
    fig.update_layout(barmode='group')
    return style(fig,'All requested rows retained — no horizon-specific model selection','Forecast horizon','Coordinate RMSE (yards)')

def save(fig,out,name):
    p=Path(out)/'figures';p.mkdir(parents=True,exist_ok=True)
    fig.write_html(p/(name+'.html'),include_plotlyjs=True,full_html=True)
    return fig
