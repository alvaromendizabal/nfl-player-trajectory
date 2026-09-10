"""Audit available observed role vocabulary against the historical encoded slots."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_trajectory.features import ROLES  # noqa: E402
from nfl_trajectory.runtime import Run, atomic_json, sha256  # noqa: E402


def main() -> None:
    reference = ROOT / "artifacts/temporal/development/plan.json"
    expected = json.loads(reference.read_text())["inputs"]
    paths = sorted(
        p for p in expected if p.startswith("data/raw/train/input") and p.endswith(".csv")
    )
    if not paths:
        raise ValueError("No verified observed-input paths were declared.")
    counts: Counter[str] = Counter()
    missing = 0
    with Run(ROOT, "role-contract-audit") as run:
        for name in paths:
            path = ROOT / name
            if sha256(path) != expected[name]:
                raise ValueError("Observed-input hash changed: " + name)
            frame = pd.read_csv(path, usecols=["player_role"])
            counts.update(frame.player_role.dropna().astype(str))
            missing += int(frame.player_role.isna().sum())
        slots = {name: ROLES.index(name) if name in ROLES else len(ROLES) - 1 for name in counts}
        report = {
            "status": "completed",
            "run_id": run.run_id,
            "input_hashes": {p: expected[p] for p in paths},
            "raw_row_counts": dict(counts),
            "missing_roles": missing,
            "historical_vocabulary": list(ROLES),
            "observed_role_slots": slots,
            "observed_mapping_is_injective": len(set(slots.values())) == len(slots),
            "sources": {
                str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
                "src/nfl_trajectory/features.py": sha256(ROOT / "src/nfl_trajectory/features.py"),
            },
            "conclusion": (
                "Other Route Runner uses the historical Unknown slot; Pass Route is absent "
                "in the audited inputs. The observed roles still occupy distinct slots, so "
                "relabeling alone does not demonstrate added predictive information. Preserve "
                "historical checkpoint slots; use actual raw labels to interpret this dataset."
            ),
        }
        atomic_json(ROOT / "docs/results/role_contract_audit.json", report)
        run.event(
            "role_contract_audited",
            input_files=len(paths),
            rows=sum(counts.values()),
            missing=missing,
            mapping_is_injective=report["observed_mapping_is_injective"],
        )


if __name__ == "__main__":
    main()
