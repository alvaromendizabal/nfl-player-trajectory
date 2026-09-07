# Start here — NFL trajectory research


## Continue an existing project after Phase 0

The first data download, audit, GitHub push, and private snapshot are complete.
Keep the existing directory. In its SageMaker terminal, run this single sequence:

```bash
cd "$HOME/nfl-player-trajectory" &&
git pull --ff-only origin main &&
python3 scripts/bootstrap.py &&
.venv/bin/nfl benchmark &&
.venv/bin/python kaggle/export.py --model role_ridge &&
.venv/bin/nfl backup &&
.venv/bin/nfl status
```

The benchmark prepares training/validation weeks, fits the role model on training
games, compares six models, and renders the report. It verifies and reuses completed
stages on repeat. The holdout is excluded. Open `notebooks/01_data_analysis.ipynb`,
`notebooks/02_motion_benchmarks.ipynb`, and `artifacts/benchmark/report.html`.
Notebook 00 remains an orientation notebook; running it is not a prerequisite for
01/02. The two research notebooks read the computed results rather than retraining.

The steps below describe a first installation on a new machine.

The work in this phase establishes trustworthy inputs, scoring, recovery, and
development practices. It does not launch neural-network training.

## 1. Open the AWS space

In SageMaker Studio, use **US West (Oregon), us-west-2**, the existing domain that
contains your other projects, and the private space **nfl-trajectory-dev**
(display name **NFL Trajectory Research**).

The space has been created with **ml.t3.large**, **50 GB** storage, and a
**60-minute idle timeout**. Start the space, then open JupyterLab. Starting it
incurs compute charges; storage and S3 are billed separately. Stop the app when
finished. Do not delete the space. A GPU is unnecessary for this phase.

## 2. Upload and extract

Upload `nfl-player-trajectory.zip` using the JupyterLab file browser. In a terminal:

```bash
cd "$HOME"
python3 -m zipfile -e nfl-player-trajectory.zip "$HOME"
cd "$HOME/nfl-player-trajectory"
```

This is the initial installation. Do not re-extract over later edits; use Git to
update the normal files in subsequent phases.

## 3. Build and verify the environment

```bash
cd "$HOME/nfl-player-trajectory"
python3 scripts/bootstrap.py
```

Wait for `bootstrap_completed`. This creates an isolated Python environment,
registers the notebook kernel, runs checks and tests, executes the orientation
notebook, and runs the synthetic pipeline twice to exercise reuse. The command
prints timestamps, elapsed time, and heartbeats. It does not exit your shell.

## 4. Authenticate Kaggle

```bash
cd "$HOME/nfl-player-trajectory"
.venv/bin/python scripts/authenticate.py
```

Open the Kaggle link printed in the terminal, sign in to the account that accepted
the competition rules, and approve access. Kaggle then displays a one-time
verification code. Paste that code into the waiting terminal and press Enter.
This remote-terminal flow uses browser approval; you do not create an API token.

The official Kaggle SDK saves the sign-in in `~/.kaggle/credentials.json` outside
the project. Later runs verify and reuse it, refreshing access when needed.
Approval links and codes are not written to project logs. The script prints UTC
timestamps, elapsed time, and a heartbeat while waiting. Keep the terminal open
until approval completes. If interrupted before sign-in is saved, rerun the same
command for a new link; completed project work remains reusable.

To deliberately sign in again, use `scripts/authenticate.py --force` with the same
Python executable. The equivalent official CLI command for initial sign-in is:

```bash
.venv/bin/kaggle auth login --no-launch-browser
```

See [Kaggle's official authentication documentation](https://www.kaggle.com/docs/api).

## 5. Download and audit the official data

```bash
cd "$HOME/nfl-player-trajectory"
.venv/bin/nfl download
```

Wait for `completed`. Then:

```bash
.venv/bin/nfl audit
```

The published inventory is 49 files, approximately 865 MB. The code queries the live
inventory instead of hardcoding a file count. Expect a few minutes depending on
network and disk performance. Each file and weekly audit produces progress events.
If either command stops, retain the directory and rerun that same command.

If authentication fails, check that you signed in to the account that accepted
the **Prediction** competition rules. Do not repeatedly create new environments.

## 6. Save a recovery snapshot

```bash
.venv/bin/nfl backup
.venv/bin/nfl status
```

The supplied ignored `aws.local.json` selects your new private bucket. The final
backup event prints a `snapshots/<hash>.json` identifier. Save it; it identifies the
complete snapshot. `artifacts/last_backup.json` also records it. A failed upload never
publishes a completed snapshot. Only completed outputs are reused during retry.

## 7. Review the notebook

Open `notebooks/00_project_readiness.ipynb`, select **Python (NFL Trajectory)**, and
Run All. Read the metric explanation and the synthetic trajectory diagram, then
review the actual audit summary and split counts after Step 5. The synthetic demo
illustrates behavior; it is not an NFL evaluation score.

The executed notebook from the quality gate is also in `artifacts/notebooks/`.
Open `artifacts/demo/trajectory.html` in a browser for the interactive offline figure.

## 8. Create the public GitHub repository

The linked GitHub connection cannot create repositories. In [GitHub New Repository](https://github.com/new):

- Owner: `alvaromendizabal`
- Name: `nfl-player-trajectory`
- Visibility: **Public**
- Description: `NFL player trajectory prediction with temporal validation, motion baselines, interpretable visualizations, and reproducible AWS experiments.`
- Leave **Add README**, **.gitignore**, and **license** unchecked; this project already includes them.

Then run:

```bash
cd "$HOME/nfl-player-trajectory"
.venv/bin/python scripts/initialize_git.py
git remote add origin https://github.com/alvaromendizabal/nfl-player-trajectory.git
GH_BROWSER=true gh auth login --hostname github.com --git-protocol https --web --scopes workflow
gh auth setup-git --hostname github.com
git push -u origin main
```

The initializer refuses an existing Git repository and commits an explicit allowlist
of source, notebooks, tests, documentation, and CI. It excludes private configuration
and generated data. Authentication for `git push` is separate from the ChatGPT GitHub
connection. Install the [official GitHub CLI](https://cli.github.com/) if `gh` is unavailable.
The login command prints a device code. Open https://github.com/login/device in your
signed-in browser, enter that code on GitHub, and authorize the CLI. Saved credentials
support subsequent pushes. Never enter an account password at a Git password prompt
or put a token in a remote URL. For an existing repository, skip initialization and
remote creation; use its existing commit history.
Creating the empty repository and sharing its link also allows the next phase to use
the linked GitHub connection for repository changes.

## 9. Confirm the handoff

The Phase 0 gate is complete when:

- `bootstrap_completed` appears and GitHub Actions passes after push.
- `artifacts/audit_summary.json` reports `passed` on actual NFL data.
- `artifacts/game_splits.csv` exists and has train, validation, and holdout games.
- `nfl backup` finishes and prints a snapshot manifest.
- Notebook 00 displays the actual audit results without errors.

Send the final `nfl status` output and repository link. Do not send data or tokens.
Next is **Phase 1: real-data EDA, field animations, split review, and scored physical
baselines**. Subsequent changes use feature branches, tested PRs, and documented merges.

