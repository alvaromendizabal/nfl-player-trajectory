# Start here

## Review without private data

Open [the Research Review](research/RESEARCH_REVIEW.ipynb). Its inline Plotly figures are generated exclusively from the public JSON aggregates in `research/evidence/studies.json`. The notebook and HTML export are evidence views, not an inference service.

From the repository root, validate the frozen published files with the Python standard library:

```bash
python research/publication/validate.py
```

Open the notebook to explore the public evidence using an environment with Plotly. The optional `python research/publication/render.py` also renders static fallbacks and therefore requires Matplotlib. Rerendering with a different library environment can legitimately change derived file bytes; the frozen publication hash manifest must not be silently rewritten to disguise that difference.

The renderer executes only the small, supplied aggregate-review notebook. It imports no experimental model code and does not request AWS credentials. The manifest validator is a standard-library check, not an assertion that every experiment was reproduced.

## Inspect the forecasting implementation

The maintained implementation is in `src/nfl_trajectory/`. Original project notebooks remain under `notebooks/`. The earlier detailed execution guide is preserved at [docs/archive/START_HERE.md](docs/archive/START_HERE.md); its old commands describe historical experiments and must not be run blindly to review this publication.

## Inspect the manual feature research

[research/workspace/README.md](research/workspace/README.md) indexes the original named experiment packages. These are preserved source mirrors, not a reorganized training application. Do not assume moving them changes the hardcoded paths or the source fingerprints in their private contracts. Public copies intentionally omit `input_contract.json`, fitted objects, raw competition files, row-level results, and private recovery receipts.

The owner continues running the original folders in the existing SageMaker workspace. Publication does not replace those folders or invalidate their checkpoints. Reproducing a historical experiment on another machine requires authorized data and the corresponding private input contract, source revision, and environment; none are inferred from a public score.

## Publication is not deployment

A GitHub merge changes the repository. It does not itself update the existing AWS checkout, retrain a model, or submit to Kaggle. The owner-side publication helper verifies the merged commit in a separate worktree. Active research contracts require the original checkout to remain at their pinned Git HEAD, so that checkout is intentionally preserved while the publication worktree displays the merged revision. No private contract is rewritten to bypass provenance checks. Private artifacts stay local/S3.
