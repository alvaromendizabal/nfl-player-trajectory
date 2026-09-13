"""Plotly reports: recorded NFL aggregates or explicitly identified test fixtures."""
from pathlib import Path
import json
import plotly.graph_objects as go

LABELS={'control':'Preserved control','direct_state':'Direct state (stopped)',
        'pass_axis':'Pass-axis state','pass_axis_history':'Pass-axis + history'}
ROLE={'role_0':'Targeted receiver','role_1':'Defensive coverage','role_2':'Passer','role_3':'Other route runner'}

def load(path):
    p=Path(path)
    if not p.is_file():raise FileNotFoundError(f'{p.name} is missing. Run its preceding stage; no placeholder result is used.')
    return json.loads(p.read_text())

def style(fig,title,y='Coordinate RMSE (yards)'):
    fig.update_layout(title=title,height=470,font={'size':14},margin={'l':75,'r':35,'t':90,'b':105},
                      legend={'orientation':'h','y':1.13},yaxis_title=y)
    return fig

def previous_folds(kit):
    r=load(Path(kit)/'evidence/round5_summary.json');fig=go.Figure()
    for a in ('control','direct_state'):
        fig.add_bar(name=LABELS[a],x=[f"Fold {f['fold']}" for f in r['fold_results']],
                    y=[f['metrics'][a] for f in r['fold_results']])
    return style(fig,'Round 5 · direct-state gain did not replicate')

def previous_horizon(kit):
    r=load(Path(kit)/'evidence/round5_summary.json')['pools']['later_folds_only'];fig=go.Figure()
    for a in ('control','direct_state'):
        rows=[v for v in r['slices'] if v['arm']==a and v['slice'] in ('first_second','after_first_second')]
        fig.add_bar(name=LABELS[a],x=[v['slice'].replace('_',' ') for v in rows],y=[v['rmse'] for v in rows])
    return style(fig,'Round 5 · horizon errors, later folds only')

def previous_roles(kit):
    r=load(Path(kit)/'evidence/round5_summary.json')['pools']['later_folds_only'];fig=go.Figure()
    for a in ('control','direct_state'):
        rows=[v for v in r['slices'] if v['arm']==a and v['slice'] in ROLE]
        fig.add_bar(name=LABELS[a],x=[ROLE[v['slice']] for v in rows],y=[v['rmse'] for v in rows])
    return style(fig,'Round 5 · role slices are diagnostics, not routing rules')

def populations(out):
    r=load(Path(out)/'preflight.json');rows=[f for f in r['fold_populations'] if f['fold'] in (2,3)];fig=go.Figure()
    for k,n in (('training_rows','Training'),('evaluation_rows','Evaluation')):
        fig.add_bar(name=n,x=[f"Fold {f['fold']}" for f in rows],y=[f[k] for f in rows])
    return style(fig,'Frozen later-fold populations','Forecast rows')

def support(out):
    r=load(Path(out)/'smoke.json')
    f=go.Figure(go.Bar(x=r['support_names'],y=r['support_fraction']))
    f.update_yaxes(range=[0,1]);return style(f,'32 training plays · measurement support','Observed fraction (player/play weighted)')

def fold_metrics(out):
    reports=[load(Path(out)/f'fold_{i}/summary.json') for i in (2,3) if (Path(out)/f'fold_{i}/summary.json').exists()]
    if not reports:raise FileNotFoundError('No completed fold result')
    f=go.Figure()
    for a in ('control','pass_axis','pass_axis_history'):
        f.add_bar(name=LABELS[a],x=[f"Fold {r['fold']}" for r in reports],y=[r['metrics'][a] for r in reports])
    return style(f,'Round 6 · matched feature comparisons by fold')

def contrasts(out):
    r=load(Path(out)/'summary.json');c=r['contrasts'];labels=[LABELS[z['treatment']]+' − '+LABELS[z['control']] for z in c]
    f=go.Figure()
    for z,label in zip(c,labels):
        # Percentile intervals need not contain the point estimate.
        f.add_scatter(x=[z['adjusted_low'],z['adjusted_high']],y=[label,label],mode='lines',showlegend=False)
        f.add_scatter(x=[z['delta_rmse']],y=[label],mode='markers',showlegend=False)
    f.add_vline(x=0,line_dash='dot');f.update_xaxes(title='RMSE difference; negative favors the treatment')
    return style(f,'Later-fold pooled effects · adjusted exploratory intervals','')

def horizon(out):
    rows=[load(Path(out)/f'fold_{i}/summary.json') for i in (2,3) if (Path(out)/f'fold_{i}/summary.json').exists()]
    if not rows:raise FileNotFoundError('No completed fold')
    f=go.Figure()
    for a in ('control','pass_axis','pass_axis_history'):
        vals=[]
        for sl in ('first_second','after_first_second'):
            ss=[v for r in rows for v in r['slices'] if v['arm']==a and v['slice']==sl]
            vals.append((sum(v['sse'] for v in ss)/(2*sum(v['rows'] for v in ss)))**.5 if ss else None)
        f.add_bar(name=LABELS[a],x=['First second','After first second'],y=vals)
    return style(f,'Round 6 · pooled horizon error (no forecast rows removed)')

def retained(out):
    rows=[load(Path(out)/f'fold_{i}/summary.json') for i in (2,3) if (Path(out)/f'fold_{i}/summary.json').exists()]
    f=go.Figure()
    for a in ('pass_axis','pass_axis_history'):
        f.add_bar(name=LABELS[a],x=[f"Fold {r['fold']}" for r in rows],y=[r['retained_columns'][a] for r in rows])
    return style(f,'Training-only constant-column screen','Retained total columns (not importance)')

def save(fig,out,name):
    folder=Path(out)/'figures';folder.mkdir(parents=True,exist_ok=True)
    fig.write_html(folder/(name+'.html'),include_plotlyjs='directory',full_html=True)
    return fig
