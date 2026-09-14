"""Plotly diagnostics. Source data are measured aggregates, never fabricated scores."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

LABELS = {'control':'Control — preserved', 'arrival':'Arrival — preserved',
          'arrival_origin':'Earlier-origin mixture — stopped',
          'velocity':'Required-velocity response', 'acceleration':'Required-acceleration response',
          'receiver':'Receiver-relative velocity', 'turn':'Turning arc', 'brake':'Braking / speed change',
          'orientation':'Orientation / velocity mismatch'}


def finish(fig, title, subtitle=''):
    fig.update_layout(title={'text':title+('<br><sup>'+subtitle+'</sup>' if subtitle else '')},
                      height=480, margin=dict(l=80,r=35,t=95,b=85),
                      font=dict(size=13), legend_title_text='', hovermode='closest')
    return fig


def prior_scores(summary):
    df=pd.DataFrame([{'arm':LABELS.get(a,a),'RMSE':v} for a,v in summary['pooled_rmse'].items()])
    return finish(px.bar(df,x='RMSE',y='arm',orientation='h',text='RMSE'),
                  'Round 1: measured pooled coordinate RMSE',
                  f"{summary['evaluation_rows']:,} selected rows / {summary['evaluation_games']} games — not Kaggle")


def prior_fold_gains(summary):
    rows=[]
    for d in summary['comparisons']:
        rows.append({'fold':str(d['fold']),'comparison':d['comparison'],'delta':d['delta_rmse'],
                     'plus':d['ci_high']-d['delta_rmse'],'minus':d['delta_rmse']-d['ci_low']})
    df=pd.DataFrame(rows)
    fig=px.scatter(df,x='fold',y='delta',color='comparison',error_y='plus',error_y_minus='minus')
    fig.add_hline(y=0,line_dash='dash')
    return finish(fig,'Round 1: fold stability','Negative difference favors treatment; ordinary 95% paired-game intervals')


def contrast_intervals(summary):
    rows=[]
    for d in summary['decisions']:
        rows.append({'comparison':d['arm']+' minus '+d['baseline'],'delta':d['delta_rmse'],
                     'plus':max(0,d['simultaneous_high']-d['delta_rmse']),
                     'minus':max(0,d['delta_rmse']-d['simultaneous_low']),
                     'direction':d['direction'],'decision':d['decision'],
                     'lower':d['simultaneous_low'],'upper':d['simultaneous_high']})
    df=pd.DataFrame(rows)
    fig=px.scatter(df,x='delta',y='comparison',color='direction',error_x='plus',error_x_minus='minus',
                   hover_data=['decision','lower','upper'])
    fig.add_vline(x=0,line_dash='dash')
    return finish(fig,summary['phase'].title()+': paired family contribution',
                  'Nine-comparison-adjusted intervals; negative addition helps, positive removal shows a cost')


def fold_scores(summary):
    df=pd.DataFrame(summary['fold_metrics']);df['fold']=df['fold'].astype(str)
    fig=px.line(df,x='fold',y='rmse',color='arm',markers=True,hover_data=['rows'])
    return finish(fig,'Arrival components across reused chronological folds','Same training rows, penalty and complete evaluation rows')


def support_bars(summary):
    df=pd.DataFrame(summary['training_only_support'])
    if df.empty:
        raise ValueError('No first-fold training support to plot')
    df['group']=df['family']+' / '+df['role']+' / w'+df['window'].astype(str)
    fig=px.bar(df,x='available_fraction',y='group',orientation='h',hover_data=['players','supported'])
    fig.update_xaxes(range=[0,1],tickformat='.0%')
    return finish(fig,'32-play smoke: observed-input support','Training-player trajectories only; a support check, not measured predictive value')


def support_heatmap(summary):
    df=pd.DataFrame(summary['training_only_support'])
    if df.empty:
        raise ValueError('No first-fold training support to plot')
    df['family_window']=df['family']+' / w'+df['window'].astype(str)
    pivot=df.pivot(index='role',columns='family_window',values='mean_adjacent_support')
    fig=px.imshow(pivot,aspect='auto',text_auto='.2f',zmin=0,zmax=1)
    return finish(fig,'Motion-feature support: complete prepared subset',
                  'Only first-fold TRAIN games; mean adjacent-observation support, not feature selection on validation')


def motion_scores(summary):
    rows=[{'arm':a,'RMSE':v} for a,v in summary['pooled_rmse'].items() if a!='control']
    return finish(px.bar(pd.DataFrame(rows),x='RMSE',y='arm',orientation='h',text='RMSE'),
                  'New motion families versus preserved arrival control','No ensemble; each family is tested separately')


def horizon_scores(summary):
    df=pd.DataFrame([r for r in summary['slices'] if r['slice'] in ('first_second','after_first_second') and r['arm']!='control'])
    agg=df.groupby(['arm','slice'],as_index=False)[['sse','rows']].sum()
    agg['RMSE']=np.sqrt(agg.sse/(2*agg.rows))
    fig=px.bar(agg,x='slice',y='RMSE',color='arm',barmode='group',hover_data=['rows'])
    return finish(fig,'Does the feature help the difficult horizons?',
                  'Pooled from squared errors and row counts — not an average of fold RMSEs')


def show_save(fig, output: Path, name: str):
    output.mkdir(parents=True,exist_ok=True)
    fig.show()
    fig.write_html(output/(name+'.html'),include_plotlyjs='directory',full_html=True)
    return output/(name+'.html')
