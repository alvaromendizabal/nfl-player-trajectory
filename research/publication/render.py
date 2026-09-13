"""Execute the public aggregate-review notebook with a minimal, recorded renderer.

Only the curated review notebook is executed. No experimental source is imported.
The execution engine is plain Python, not a claimed Jupyter kernel execution.
"""
from pathlib import Path
import contextlib
import base64
import io
import json

import plotly.io as pio

ROOT = Path(__file__).resolve().parents[2]


def static_png(fig):
    """Static fallback from the same public trace values; Plotly remains inline."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    chart, ax = plt.subplots(figsize=(12, 5.5))
    traces = list(fig.data)
    if traces and traces[0].type == 'bar':
        categories = list(dict.fromkeys(str(x) for t in traces for x in t.x))
        width = .8 / max(1, len(traces))
        for j, t in enumerate(traces):
            index = [categories.index(str(x)) + (j-(len(traces)-1)/2)*width for x in t.x]
            ax.bar(index, list(t.y), width=width, label=t.name)
        ax.set_xticks(range(len(categories)), categories, rotation=35, ha='right')
        if len(traces) > 1:
            ax.legend(fontsize=8)
    else:
        for t in traces:
            labels = list(t.y)
            low = list(t.error_x.arrayminus) if t.error_x.arrayminus is not None else [0]*len(labels)
            high = list(t.error_x.array) if t.error_x.array is not None else [0]*len(labels)
            ax.errorbar(list(t.x), range(len(labels)), xerr=[low,high], fmt='o', capsize=3)
            ax.set_yticks(range(len(labels)), labels)
        ax.axvline(0, linestyle='--', linewidth=1)
    ax.set_title(fig.layout.title.text or '')
    ax.set_xlabel(fig.layout.xaxis.title.text or '')
    ax.set_ylabel(fig.layout.yaxis.title.text or '')
    chart.tight_layout()
    stream=io.BytesIO()
    chart.savefig(stream, format='png', dpi=140)
    plt.close(chart)
    return stream.getvalue()


def main():
    path = ROOT/'research/RESEARCH_REVIEW.ipynb'
    nb = json.loads(path.read_text())
    ns = {'__name__':'__publication_review__','PUBLIC_ROOT':ROOT}
    figures = []
    outputs = []

    def show(fig):
        obj = json.loads(pio.to_json(fig, remove_uids=True))
        png=static_png(fig)
        folder=ROOT/'research/figures'
        folder.mkdir(parents=True,exist_ok=True)
        (folder/f'review_{len(figures)+1}.png').write_bytes(png)
        outputs.append({'output_type':'display_data','metadata':{},'data':{
            'application/vnd.plotly.v1+json':obj,
            'image/png':base64.b64encode(png).decode(),
            'text/plain':['Interactive Plotly figure rendered from published aggregate evidence.']}})
        figures.append(fig)

    ns['show'] = show
    count = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        count += 1
        outputs.clear()
        stream = io.StringIO()
        source = ''.join(cell['source'])
        with contextlib.redirect_stdout(stream):
            exec(compile(source, 'RESEARCH_REVIEW.ipynb', 'exec'), ns)
        cell['execution_count'] = count
        cell['outputs'] = list(outputs)
        if stream.getvalue():
            cell['outputs'].insert(0, {'output_type':'stream','name':'stdout','text':stream.getvalue().splitlines(True)})
    nb['metadata']['publication_execution'] = {
        'engine':'plain-Python public-evidence renderer; not Jupyter kernel execution',
        'code_cells':count,'figures':len(figures),'private_data_used':False,'scientific_fits':0}
    path.write_text(json.dumps(nb, indent=1, allow_nan=False)+'\n')
    html=['<!doctype html><meta charset="utf-8"><title>NFL trajectory research evidence</title>',
          '<h1>NFL trajectory research evidence</h1><p>Published aggregate results only. '
          'Repeated internal game splits are not Kaggle scores.</p>']
    for i, fig in enumerate(figures):
        html.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn' if i==0 else False, div_id=f'nfl-evidence-{i}'))
    (ROOT/'research/RESEARCH_REVIEW.html').write_text('\n'.join(html))
    print(json.dumps({'status':'public_evidence_rendered','code_cells':count,'figures':len(figures),
                      'private_data_used':False,'scientific_fits':0}))


if __name__ == '__main__':
    main()
