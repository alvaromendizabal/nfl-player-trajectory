from pathlib import Path
import nbformat as nbf
root = Path(__file__).resolve().parents[1]
nb=nbf.v4.new_notebook()
nb.metadata={'kernelspec':{'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'},'language_info':{'name':'python'},'nfl_evidence_mode':'synthetic demonstration unless verified local receipts are present'}
c=[]
def md(s: str) -> None: c.append(nbf.v4.new_markdown_cell(s))
def code(s: str) -> None: c.append(nbf.v4.new_code_cell(s))
md('''# Temporal interaction research
## Preserve *when* player relationships change

**Feature engineering is open. Predictive benefit of this prototype is not measured.**
This notebook separates three things: historical performance, verified saved-run evidence,
and feature-construction diagnostics. It launches no AWS jobs and performs no training.

The historical **0.62708** internal result lacks its exact model artifacts; it is not a
replayable release or a Kaggle score. The last recorded private Kaggle score is **0.70090**.
The objective remains approximately **0.46**, not yet achieved.

The existing coordinate-versus-velocity experiment must be inspected before another fit.
See [the bounded protocol](../docs/TEMPORAL_EDGE_PROTOCOL.md).
''')
code('''import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import FileLink, display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src/nfl_trajectory").is_dir())
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "artifacts/interaction_milestone"
OUT.mkdir(parents=True, exist_ok=True)
FIGURES = []


def show(figure):
    FIGURES.append(figure)
    payload = figure.to_plotly_json()
    payload.get("layout", {}).pop("template", None)
    display({"application/vnd.plotly.v1+json": payload}, raw=True)


print("No training or cloud mutation is performed by this notebook.")
print("Private inputs and saved errors stay under the ignored artifacts directory.")''')
md('''### 1 · The question being tested

Terminal pair geometry and fixed pooled affinities can lose the order of movement changes.
The new representation keeps **source player × destination player × observed frame × channel**.
It is a tensor interface for a future temporal interaction encoder, not a learned coverage model.

Eleven channels encode relative position and velocity, distance, closing/lateral speed,
velocity alignment, bearing/separation rates, and same-side status. Every measurement has
an explicit validity mask. Missing frames are not filled or compressed; rates require
adjacent jointly observed frames. Stationary and coincident cases remain distinguishable.

Primary evidence: [Song et al. (2026), §§3.1–3.3](https://arxiv.org/html/2603.25901v1)
and [Dutta, Yurko & Ventura](https://arxiv.org/abs/1906.11373) motivate temporal player
relationships. Their coverage labels, contextual inputs, and classification accuracy
are **not** our inference inputs or trajectory-RMSE evidence.
''')
code('''from nfl_trajectory.temporal_edges import NAMES, SCALES, build_edges

example_file = OUT / "edges_example.npz"
smoke_file = OUT / "feature_smoke.json"
if example_file.exists() and smoke_file.exists():
    smoke = json.loads(smoke_file.read_text())
    assert smoke["status"] == "training_only_feature_smoke_passed"
    assert hashlib.sha256(example_file.read_bytes()).hexdigest() == smoke["example_sha256"]
    with np.load(example_file, allow_pickle=False) as saved:
        edges = {name: saved[name].copy() for name in saved.files}
    MODE = "Training-only cache smoke: no scientific fit"
else:
    time = np.arange(20) / 10
    position = np.zeros((4, 20, 2))
    position[0, :, 0] = 40 + 4 * time
    position[0, :, 1] = 20 + np.maximum(time - 0.8, 0) * 3
    position[1, :, 0] = 44 + 2 * time
    position[1, :, 1] = 21 + np.maximum(time - 1.0, 0) * 2
    position[2, :, 0] = 46 + 3 * time
    position[2, :, 1] = 28 - 2 * time
    position[3, :, 0] = 41 + 3 * time
    position[3, :, 1] = 30 - time
    velocity = np.diff(position, axis=1, prepend=position[:, :1]) / 0.1
    velocity[:, 0] = velocity[:, 1]
    seen = np.ones((4, 20), dtype=bool)
    seen[2, 7:10] = False
    edges = build_edges(position, velocity, seen, np.array([1, 0, 0, 1]))
    smoke = None
    MODE = "SYNTHETIC mechanics demonstration: not NFL performance"

physical = edges["values"] * SCALES
print(MODE)
print("Tensor:", edges["values"].shape)
print("Total tensor/mask bytes:", sum(x.nbytes for x in edges.values()))
display(pd.DataFrame({"channel": NAMES, "physical_scale": SCALES.tolist()}))''')
md('''### 2 · Observed relationship diagnostics

These charts show the selected evidence mode in every title. Blank entries mean
unavailable measurements, not zero separation or zero motion. Player slots are
identifiers within a play, not numerical player-ID features. **No true coverage
assignment is inferred by these charts.**''')
code('''distance = np.where(edges["valid"][..., 4], physical[..., 4], np.nan)
show(go.Figure(go.Heatmap(z=distance[:, :, -1].tolist())).update_layout(
    title=MODE + " — final observed pair distances", xaxis_title="Destination slot",
    yaxis_title="Source slot", height=450))

series = go.Figure()
clock = (np.arange(distance.shape[-1]) - distance.shape[-1] + 1) / 10
for peer in range(1, min(distance.shape[0], 7)):
    series.add_scatter(x=clock.tolist(), y=distance[0, peer].tolist(),
                       mode="lines+markers", connectgaps=False, name=f"Slot 0 to {peer}")
show(series.update_layout(title=MODE + " — spacing through the observed window",
                          xaxis_title="Seconds before observed cutoff",
                          yaxis_title="Separation (yards)", height=450))''')
code('''validity = edges["valid"].sum(axis=(0, 1, 2))
count = max(int(edges["pair_valid"].sum()), 1)
show(go.Figure(go.Bar(x=list(NAMES), y=(validity / count).tolist())).update_layout(
    title=MODE + " — channel support on jointly observed pair-frames",
    yaxis_title="Supported fraction", height=430))

mask = edges["valid"][0, 1].T.astype(int)
show(go.Figure(go.Heatmap(x=clock.tolist(), y=list(NAMES), z=mask.tolist())).update_layout(
    title=MODE + " — explicit validity for pair 0 → 1",
    xaxis_title="Seconds before observed cutoff", height=430))''')
md('''### 3 · Existing scientific run: verify, do not rerun

The AWS helper reads `nfl-motion-scientific-20260911-055510-d265d9d` only. It verifies
account and source identity, immutable hashes, all requested error rows, and the
reported coordinate RMSE. It does not launch jobs or load pickled model code.
A checkpoint-byte check is **not** an independent numerical model restoration.

When the local inspection receipt exists, the next cell recomputes the paired
whole-game bootstrap using the existing experiment's NumPy generator, seed,
10,000 resamples, and pooled-coordinate definition. Passing this statistical
check still does not establish a leaderboard score or certify inference replay.
''')
code('''inspection_path = OUT / "inspection.json"
if inspection_path.exists():
    inspection = json.loads(inspection_path.read_text())
    print("AWS observation:", inspection.get("observed_utc"))
    print("Job status:", inspection.get("job_status", "not established"))
    print("Evidence status:", inspection["status"])
    if inspection["status"] == "saved_errors_and_checkpoint_bytes_verified":
        totals = inspection["recomputed"]
        games = pd.DataFrame(totals["per_game"]).sort_values("game_id")
        indices = np.random.default_rng(2026).integers(0, len(games), (10000, len(games)))
        counts = games["rows"].to_numpy()[indices].sum(axis=1)
        control = np.sqrt(games["control_sse"].to_numpy()[indices].sum(axis=1) / (2 * counts))
        velocity = np.sqrt(games["velocity_sse"].to_numpy()[indices].sum(axis=1) / (2 * counts))
        low, high = np.quantile(velocity - control, [0.025, 0.975])
        reported = inspection["reported_evaluation"]["paired_game_bootstrap"]
        interval_matches = bool(np.allclose(
            [low, high], [reported["delta_ci95_low"], reported["delta_ci95_high"]], atol=1e-6))
        passed = totals["relative_gain"] >= 0.01 and high < 0 and interval_matches
        print("Recomputed coordinate RMSE:", totals["control_rmse"], totals["velocity_rmse"])
        print("Paired 95% RMSE-difference interval:", float(low), float(high))
        print("Saved interval agrees:", interval_matches, "Statistical criterion passes:", passed)
        print("Numerical checkpoint restoration is still required before model promotion.")
        delta = np.sqrt(games["velocity_sse"] / (2 * games["rows"]))
        delta -= np.sqrt(games["control_sse"] / (2 * games["rows"]))
        plot = go.Figure(go.Bar(x=games["game_id"].astype(str).tolist(), y=delta.tolist()))
        show(plot.update_layout(
            title="Verified saved errors — per-game velocity minus control RMSE",
            xaxis_title="Evaluation game", yaxis_title="RMSE difference (yards)", height=450))
    else:
        print("No accepted comparison. Inspect the saved failure/status before any new training.")
else:
    print("No local AWS inspection receipt. Live result and new RMSE are UNKNOWN.")
    print("Run the supplied bounded helper in AWS, then rerun this notebook.")''')
md('''### 4 · Decision and the next experiment

**Do not alter the old run.** Close its result first: all 83,938 rows, both final
1,248-step arm states, independent restoration, and the frozen >=1% improvement
plus negative upper confidence-bound gate. Stop failed scientific treatments;
diagnose execution failures before compatible recovery.

For a subsequent **separately versioned** interaction study, keep the same added
edge encoder in both arms. Compare repeated terminal edge values versus the real
observed edge sequence with matching masks, model capacity, loss, data, seed,
augmentation, final EMA, and optimizer exposure. Freeze its missing-terminal
policy and benchmark training-only throughput before selecting the budget.
Do not import unavailable coverage labels or tune on an inspected holdout.

This prototype has not undergone real-data input screening, model integration,
feature-value ablation, or chronological replication. These remain explicit gates.
The notebook may show a 32-play observed-only smoke after the helper is run;
that smoke is not a scientific experiment or evidence of improved RMSE.
''')
code('''html_path = OUT / "interaction_report.html"
sections = ["<h1>NFL temporal interaction research</h1>",
            "<p>" + MODE + "</p>",
            "<p>Feature research remains open. No training was performed by this notebook.</p>"]
for index, figure in enumerate(FIGURES):
    sections.append(pio.to_html(figure, include_plotlyjs=index == 0, full_html=False))
html_path.write_text("<!doctype html><html><meta charset='utf-8'><body>" +
                     "\\n".join(sections) + "</body></html>", encoding="utf-8")
print("Saved offline interactive report:", html_path)
display(FileLink(str(html_path)))''')
nb.cells=c
nbf.write(nb,root/'notebooks/03_interaction_research.ipynb')
