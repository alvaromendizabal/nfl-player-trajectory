"""Plotly views with explicit training-versus-evaluation provenance."""
from pathlib import Path
import json
import plotly.graph_objects as go


def load(p):return json.loads(Path(p).read_text())
def style(f,title,y):
    f.update_layout(title=title,yaxis_title=y,font=dict(size=14),height=500,
                    margin=dict(l=85,r=30,t=95,b=110),legend=dict(orientation='h',y=1.04))
    return f

def save(f,out,name):
    p=Path(out)/'figures';p.mkdir(parents=True,exist_ok=True)
    f.write_html(p/(name+'.html'),include_plotlyjs=True,full_html=True)
    f.write_json(p/(name+'.json'))
    return f

def previous_metrics(kit):
    s=load(Path(kit)/'evidence/round10_summary.json')
    x=['Preserved tree','Cartesian encoder','Goal encoder'];y=[s['metrics'][k] for k in ['preserved_tree','cartesian','goal']]
    return style(go.Figure(go.Bar(x=x,y=y,text=[f'{v:.6f}' for v in y],textposition='outside')),
                 'Round 10 · completed internal evaluation (5,886 rows)','Coordinate RMSE, yards')

def previous_contrast(kit):
    c=load(Path(kit)/'evidence/round10_summary.json')['contrast'];d=c['delta_rmse']
    f=go.Figure(go.Scatter(x=['Goal minus Cartesian'],y=[d],mode='markers',
       error_y=dict(type='data',symmetric=False,array=[c['high95']-d],arrayminus=[d-c['low95']])))
    f.add_hline(y=0,line_dash='dash')
    return style(f,'Round 10 · saved paired-game 95% interval crosses zero','RMSE difference, yards')

def coverage(out):
    s=load(Path(out)/'coverage_summary.json')
    f=go.Figure(go.Bar(x=['Used in existing training subset','Other observed plays in same training games'],
        y=[s['selected_training_plays'],s['eligible_observed_plays']-s['selected_training_plays']]))
    return style(f,'Training coverage · identity-only inventory; unused labels not verified','Distinct plays')

def support(out):
    s=load(Path(out)/'feature_summary.json')['channel_statistics']
    return style(go.Figure(go.Bar(x=[q['name'] for q in s],y=[q['support_fraction'] for q in s])),
      'Candidate support · training inputs only; denominator includes empty frame slots','Fraction of player/frame slots')

def tails(out):
    s=load(Path(out)/'feature_summary.json')['channel_statistics']
    f=go.Figure()
    # Mixed physical units are not directly ranked; division by fixed declared scale puts them in model units.
    from new_features import SCALES
    for name in ['rms','p99','max_abs']:
        f.add_bar(name=name,x=[q['name'] for q in s],y=[None if q[name] is None else q[name]/SCALES[i] for i,q in enumerate(s)])
    return style(f,'Candidate tails · fixed-scaled magnitudes, not feature importance','Scaled feature value')

def training_vs_evaluation(out):
    s=load(Path(out)/'training_audit.json');f=go.Figure();arms=['cartesian','goal']
    f.add_bar(name='Final-checkpoint training error (new audit)',x=arms,y=[s['metrics'][k]['rmse'] for k in arms])
    f.add_bar(name='Historical evaluation score (not rerun)',x=arms,y=[s['historical_evaluation_metrics'][k] for k in arms])
    return style(f,'Fit and generalization · different populations, not an improvement comparison','Coordinate RMSE, yards')

def horizon_errors(out):
    s=load(Path(out)/'training_audit.json');f=go.Figure()
    for arm in ['zero_residual_reference','cartesian','goal']:
        rows=[x for x in s['slices'] if x['arm']==arm and 'second' in x['slice']]
        f.add_bar(name=arm,x=[x['slice'] for x in rows],y=[x['rmse'] for x in rows])
    return style(f,'Training-only horizon error · zero reference is not a fitted model','Coordinate RMSE, yards')

def learning(kit):
    f=go.Figure()
    for arm in ['cartesian','goal']:
        s=load(Path(kit)/f'evidence/round10_models_{arm}_complete.json')
        f.add_scatter(name=arm,x=list(range(1,len(s['training_epoch_objectives'])+1)),y=s['training_epoch_objectives'],mode='lines+markers')
    f.update_layout(xaxis_title='Completed epoch')
    return style(f,'Historical online training objectives · measured during updates, not final fit error','Recorded training objective')

def gates(out):
    specs=[('Preflight','preflight.json','training_review_preflight_passed'),('Coverage','coverage_summary.json','training_coverage_complete'),
       ('Smoke','smoke.json','motion_receiver_smoke_passed'),('Features','feature_summary.json','motion_receiver_features_ready'),
       ('Frozen training audit','training_audit.json','frozen_training_audit_complete'),('Replay','replay.json','training_review_replay_exact')]
    vals=[];text=[]
    for name,file,want in specs:
        p=Path(out)/file;status=load(p).get('status') if p.is_file() else 'missing'
        vals.append(int(status==want));text.append(status)
    f=go.Figure(go.Bar(x=[s[0] for s in specs],y=vals,text=text,textposition='outside'))
    f.update_yaxes(range=[0,1.5],tickvals=[0,1],ticktext=['Not verified','Verified'])
    return style(f,'Milestone readiness · software evidence, not predictive acceptance','Receipt status')
