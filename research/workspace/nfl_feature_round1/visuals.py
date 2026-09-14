"""Plotly diagnostics from actual receipts or explicitly synthetic tests."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def show_save(fig,path):
    """One shared local Plotly JS bundle keeps exports usable offline."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.update_layout(font=dict(family='Arial',size=14),margin=dict(l=60,r=40,t=85,b=65),
                      title=dict(x=0.03),legend=dict(orientation='h',yanchor='bottom',y=1.03),
                      height=fig.layout.height or 480)
    fig.write_html(path,include_plotlyjs='directory',full_html=True)
    fig.show(renderer="plotly_mimetype")
    return path


def old_error_slices(receipts):
    values=receipts['latest_inspect-result.json']['reported_evaluation']['velocity_slices']
    frame=pd.DataFrame([v for v in values if v['dimension']=='forecast_second'])
    frame['sse']=2*frame['rows']*frame['coordinate_rmse_yards']**2
    frame['squared_error_share']=frame['sse']/frame.sse.sum()
    frame['row_share']=frame['rows']/frame.rows.sum()
    return frame


def error_mass_figure(frame):
    tidy=frame.melt(id_vars=['value'],value_vars=['row_share','squared_error_share'],var_name='quantity',value_name='share')
    fig=px.bar(tidy,x='value',y='share',color='quantity',barmode='group',
               title='Recovered velocity experiment: later rows carry most squared error',
               labels={'value':'Forecast-second bin','share':'Share of total'})
    fig.update_yaxes(tickformat='.0%');return fig


def origins_figure(summary):
    d=pd.DataFrame([{'offset_frames':int(k),**v} for k,v in summary['offsets'].items()])
    d['postthrow_rows']=d['rows']-d['prethrow_rows']
    tidy=d.melt(id_vars='offset_frames',value_vars=['postthrow_rows','prethrow_rows'],value_name='target_rows',var_name='source')
    return px.bar(tidy,x='offset_frames',y='target_rows',color='source',title='Constructed target rows by earlier origin',
                  labels={'offset_frames':'Frames before the original pass cutoff'})


def horizon_figure(data):
    # Count rows; this is data-support evidence, not an error chart.
    frame=pd.DataFrame({'forecast_seconds':data['keys'][:,3]/10,'origin_frames':data['offset'].astype(str)})
    return px.histogram(frame,x='forecast_seconds',color='origin_frames',barmode='group',nbins=40,
                        title='Training-side horizon support after origin augmentation')


def feature_support_figure(data,names):
    rows=[]
    for offset in np.unique(data['offset']):
        x=data['X'][data['offset']==offset]
        for j,name in enumerate(names):
            rows.append({'origin_frames':int(offset),'feature':name,'finite_fraction':float(np.isfinite(x[:,j]).mean()),
                         'nonzero_fraction':float((np.abs(x[:,j])>1e-9).mean()),'std':float(x[:,j].std())})
    d=pd.DataFrame(rows)
    # A zero rate may be a meaningful role gate, not missing data.
    selected=d[d.feature.str.contains('required_|receiver_velocity_gap')]
    pivot=selected.pivot(index='feature',columns='origin_frames',values='nonzero_fraction')
    fig=px.imshow(pivot,aspect='auto',title='Arrival-feature activation by origin (zeros include role gates)',
                  labels=dict(x='Origin offset in frames',y='Feature',color='Nonzero fraction'))
    fig.update_layout(height=680)
    return fig,d


def pooled_figure(summary):
    d=pd.DataFrame([{'arm':a,'rmse':v} for a,v in summary['pooled_rmse'].items()])
    return px.bar(d,x='arm',y='rmse',text_auto='.5f',title='Training-side diagnostic only: pooled coordinate RMSE',
                  labels={'rmse':'Coordinate RMSE (yards)'})


def folds_figure(summary):
    d=pd.DataFrame(summary['fits'])
    return px.line(d,x='fold',y='rmse',color='arm',markers=True,title='Fixed-ridge feature comparisons across chronological folds',
                   labels={'rmse':'Coordinate RMSE (yards)'})


def intervals_figure(summary):
    d=pd.DataFrame(summary['decisions'])
    fig=go.Figure()
    for _,r in d.iterrows():
        # Bootstrap CI need not contain the observed point; draw endpoints as a line
        # rather than assuming positive error-bar extents.
        fig.add_trace(go.Scatter(x=[r.ci_low,r.ci_high],y=[r.comparison,r.comparison],mode='lines',
                                 name=r.comparison,showlegend=False))
        fig.add_trace(go.Scatter(x=[r.delta_rmse],y=[r.comparison],mode='markers',showlegend=False,
                                 name='Observed delta',hovertemplate='%{x:.6f} yards<extra></extra>'))
    fig.add_vline(x=0,line_dash='dash');fig.update_layout(title='Paired game-bootstrap 95% intervals (exploratory)',
                                                       xaxis_title='Treatment minus control RMSE; negative is better')
    return fig


def slice_figure(summary):
    d=pd.DataFrame(summary['slices'])
    return px.bar(d,x='fold',y='rmse',color='arm',facet_col='slice',barmode='group',
                  title='Short versus long horizon: no forecast rows removed',labels={'rmse':'Coordinate RMSE (yards)'})
