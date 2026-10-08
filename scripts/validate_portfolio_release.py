"""Validate curated closeout evidence and public files without private artifacts.

This complements the frozen historical publication validator. Private evidence
digests document provenance; this public check cannot authenticate their contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = (
    "README.md",
    "START_HERE.md",
    "docs/EMPLOYER_CASE_STUDY.md",
    "docs/RESULTS.md",
    "docs/REPRODUCIBILITY.md",
    "docs/PROJECT_CLOSEOUT.md",
    "docs/MODEL_CARD.md",
    "docs/DATA_CARD.md",
    "docs/CURRENT_RESEARCH_STATUS.md",
    "docs/SOURCES.md",
    "docs/results/project_closeout.json",
    "docs/assets/project-overview.svg",
    "docs/assets/demo-preview.svg",
    "src/nfl_trajectory/portfolio_demo.py",
    "scripts/run_portfolio_demo.py",
    "scripts/validate_portfolio_release.py",
    "tests/test_portfolio_demo.py",
    "tests/test_portfolio_release.py",
    ".github/workflows/portfolio.yml",
)
FRONT_PAGES = ("README.md", "START_HERE.md", "docs/EMPLOYER_CASE_STUDY.md")
BLOCKED_SUFFIXES = {
    ".pt",
    ".pth",
    ".pkl",
    ".pickle",
    ".npz",
    ".npy",
    ".parquet",
    ".zip",
    ".gz",
    ".bin",
    ".pem",
    ".key",
    ".pyz",
}
BLOCKED_NAMES = {
    "kaggle.json",
    "aws.local.json",
    "input_contract.json",
    "remote_journal.json",
    "model_manifest.json",
    "return_manifest.json",
    "execution_ledger.json",
}
SECRET_PATTERN = re.compile(
    r"(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{25,}"
    r"|github_pat_[A-Za-z0-9_]{30,}|KGAT_[A-Za-z0-9_-]{25,}"
    r"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    r"|[?&](?:X-Amz-Signature|X-Amz-Credential|X-Goog-Signature)="
)
PRIVATE_LOCATION = re.compile(r"/home/" r"sagemaker-user/|s3://[a-z0-9][a-z0-9.-]{2,62}/")
ASPIRATION = re.compile(
    r"\b(?:beat|surpass|chase|overtake)\b.{0,60}\b(?:winner|top score|leaderboard)\b"
    r"|\b(?:goal|target|aim)\b.{0,60}\b(?:top score|first place|leaderboard)\b"
    r"|\bclose (?:the |our )?(?:remaining )?gap\b",
    re.IGNORECASE,
)
LINK = re.compile(r"!?\[[^\]]*\]\((<[^>]+>|[^\s)]+)(?:\s+[\"'][^\"']*[\"'])?\)")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("PORTFOLIO_RELEASE: " + message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "unsafe public file path")
    target = root / path
    require(not any(p.is_symlink() for p in (target, *target.parents)), "linked public file")
    require(target.is_file(), "missing public file: " + relative)
    return target


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_evidence(root: Path) -> dict[str, Any]:
    evidence = json.loads(public_path(root, "docs/results/project_closeout.json").read_text())
    require(evidence.get("schema_version") == 1, "unsupported evidence schema")
    require(evidence.get("status") == "closed_research_project", "project is not closed")
    require(evidence.get("competition") == "nfl-big-data-bowl-2026-prediction", "competition")
    metric = evidence["metric"]
    require(
        metric["name"] == "coordinate_rmse_yards" and metric["lower_is_better"] is True,
        "coordinate metric contract",
    )
    accepted = evidence["accepted_system"]
    require(
        accepted["model_count"] == 20 and accepted["private_rmse"] == 0.46468,
        "recorded accepted system differs",
    )
    require(str(accepted["submission_ref"]) == "56928100", "recorded submission differs")
    require(
        accepted["late_submission"] is True and accepted["official_rank_claim"] is False,
        "late score must not imply official placement",
    )
    require(
        all(finite(accepted[k]) for k in ("previous_private_rmse", "private_improvement")),
        "finite private progression",
    )
    require(
        math.isclose(
            accepted["previous_private_rmse"] - accepted["private_rmse"],
            accepted["private_improvement"],
            abs_tol=1e-12,
        ),
        "private gain arithmetic",
    )
    scopes = evidence["evaluation_scopes"]
    require(isinstance(scopes, list) and len(scopes) == 3, "exactly three evaluation scopes")
    mapping = {row["id"]: row for row in scopes}
    expected = {"private_submission", "supported_oof", "complete_oof"}
    require(set(mapping) == expected, "distinct private/supported/complete scopes required")
    private = mapping["private_submission"]
    require(
        private["rmse"] == accepted["private_rmse"]
        and private["rows"] is None
        and private["games"] is None,
        "private population cannot be copied from local OOF",
    )
    for name, rows, rmse in (
        ("supported_oof", 561607, 0.4629258203901377),
        ("complete_oof", 562936, 0.5242026276721191),
    ):
        row = mapping[name]
        require(row["rows"] == rows and row["games"] == 272, "population differs: " + name)
        require(
            finite(row["rmse"]) and finite(row["sse"]) and row["sse"] >= 0,
            "finite aggregate metric: " + name,
        )
        require(
            math.isclose(row["rmse"], rmse, abs_tol=1e-12, rel_tol=0),
            "recorded OOF differs: " + name,
        )
        require(
            math.isclose(row["rmse"], math.sqrt(row["sse"] / (2 * rows)), abs_tol=1e-12, rel_tol=0),
            "OOF SSE arithmetic: " + name,
        )
    require(bool(evidence.get("scope_warning")), "evaluation scope warning missing")
    repro = evidence["reproducibility"]
    for key in (
        "public_demo_reproduces_champion",
        "champion_weights_published",
        "training_launched_by_closeout",
    ):
        require(repro[key] is False, "public/private boundary differs: " + key)
    require(
        repro["public_baseline_weights_already_available"] is True,
        "existing public baseline must be distinguished from private champion",
    )
    require("synthetic" in repro["public_demo"].lower(), "demo scope must be synthetic")
    benchmark = evidence["inference_benchmark"]
    require(
        all(
            finite(benchmark[k]) and benchmark[k] > 0
            for k in ("speedup", "legacy_seconds_per_play", "shared_seconds_per_play")
        ),
        "finite benchmark timing",
    )
    require(
        math.isclose(
            benchmark["legacy_seconds_per_play"] / benchmark["shared_seconds_per_play"],
            benchmark["speedup"],
            rel_tol=1e-12,
        ),
        "speedup arithmetic",
    )
    require(
        benchmark["plays"] == 96
        and benchmark["rows"] == 3723
        and benchmark["maximum_coordinate_difference_yards"] == 0,
        "benchmark scope",
    )
    provenance = evidence["provenance"]
    require(
        isinstance(provenance, list) and 3 <= len(provenance) <= 20,
        "bounded provenance receipt inventory",
    )
    claims = {row["claim"] for row in provenance}
    require(len(claims) == len(provenance), "duplicate provenance claim")
    require(
        {
            "recorded private submission score",
            "full and supported OOF metrics",
            "inference acceleration",
        }.issubset(claims),
        "core provenance claim missing",
    )
    public_proofs = 0
    for row in provenance:
        require(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None, "provenance digest")
        if row["visibility"] == "public historical snapshot":
            require(
                digest(public_path(root, row["evidence"])) == row["sha256"],
                "public provenance hash differs",
            )
            public_proofs += 1
        else:
            require(
                row["visibility"] == "private source; curated aggregate only",
                "private source visibility must be explicit",
            )
    require(public_proofs == 1, "one independently hash-checked public timing source")
    history = evidence["historical_comparison"]
    first, final = history["first_recorded_private_rmse"], history["final_recorded_private_rmse"]
    require(
        finite(first) and first > 0 and final == accepted["private_rmse"],
        "historical progression endpoints",
    )
    require(
        math.isclose(first - final, history["absolute_reduction"], abs_tol=1e-12)
        and math.isclose(
            100 * (first - final) / first, history["relative_reduction_percent"], abs_tol=1e-10
        ),
        "historical progression arithmetic",
    )
    historical_source = json.loads(public_path(root, history["public_source"]).read_text())
    matches = [
        row
        for row in historical_source["submissions"]
        if str(row["ref"]) == history["first_recorded_submission_ref"]
    ]
    require(
        len(matches) == 1 and matches[0]["private_rmse"] == first,
        "historical first score must match its public evidence",
    )
    return {
        "evaluation_scopes": 3,
        "local_rmse_arithmetic_checks": 2,
        "public_provenance_hash_checks": public_proofs,
        "private_source_contents_verified": False,
    }


def validate_links(root: Path, path: Path, text: str) -> int:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    checked = 0
    for match in LINK.finditer(text):
        reference = match.group(1).strip("<>")
        parsed = urlsplit(reference)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        destination = (path.parent / unquote(parsed.path)).resolve()
        require(destination.is_relative_to(root.resolve()), "Markdown link escapes repository")
        require(
            destination.exists(),
            "broken local Markdown link in " + str(path.relative_to(root)) + ": " + reference,
        )
        checked += 1
    return checked


def validate_svg(path: Path) -> None:
    tree = ElementTree.fromstring(path.read_text())
    require(tree.tag.endswith("}svg") or tree.tag == "svg", "invalid SVG asset")
    for node in tree.iter():
        require(node.tag.split("}")[-1] not in ("script", "foreignObject"), "active SVG content")
        for key, value in node.attrib.items():
            require(not key.lower().startswith("on"), "SVG event handler")
            if key.split("}")[-1] == "href":
                require(value.startswith("#"), "external SVG reference")


def validate_files(root: Path, files: tuple[str, ...] = PUBLIC_FILES) -> dict[str, int]:
    require(len(files) == len(set(files)), "duplicate release allowlist entry")
    links = 0
    for name in files:
        path = public_path(root, name)
        require(
            path.suffix.lower() not in BLOCKED_SUFFIXES
            and path.name.lower() not in BLOCKED_NAMES
            and not path.name.startswith(".env"),
            "private artifact in release: " + name,
        )
        text = path.read_text(encoding="utf-8")
        require(SECRET_PATTERN.search(text) is None, "credential or signed URL in " + name)
        require(PRIVATE_LOCATION.search(text) is None, "private owner/cloud location in " + name)
        if path.suffix == ".md":
            links += validate_links(root, path, text)
        if path.suffix == ".svg":
            validate_svg(path)
        if name in FRONT_PAGES:
            require(ASPIRATION.search(text) is None, "competitive aspiration in front page " + name)
    return {"curated_files_checked": len(files), "local_markdown_links_checked": links}


def validate_demo(directory: Path) -> dict[str, int]:
    metrics = json.loads((directory / "metrics.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    require(
        metrics.get("synthetic_only") is True and manifest.get("synthetic_only") is True,
        "demo must declare synthetic-only inputs",
    )
    require(metrics.get("training_fits") == 0, "public demo cannot fit private models")
    expected = {
        "dashboard.html",
        "trajectories.svg",
        "metrics.json",
        "predictions.csv",
        "observations.csv",
        "labels.csv",
        "splits.csv",
    }
    records = manifest["files"]
    require(
        len(records) == len(expected) and {r["file"] for r in records} == expected,
        "exact deterministic demo output inventory",
    )
    for record in records:
        path = public_path(directory, record["file"])
        require(
            path.stat().st_size == record["bytes"] and digest(path) == record["sha256"],
            "demo output hash mismatch: " + record["file"],
        )
    for name in ("dashboard.html", "trajectories.svg"):
        content = (directory / name).read_text()
        require("synthetic" in content.lower(), "synthetic figure label missing")
        require(
            re.search(r"(?:src|href)=[\"']https?://", content, re.IGNORECASE) is None,
            "demo rendering depends on remote assets",
        )
    validate_svg(directory / "trajectories.svg")
    return {"synthetic_demo_files_verified": len(records)}


def validate_preview(root: Path, demo_output: Path | None = None) -> dict[str, Any]:
    evidence = json.loads(public_path(root, "docs/results/project_closeout.json").read_text())
    preview = evidence["public_demo_preview"]
    require(preview["path"] == "docs/assets/demo-preview.svg", "declared demo preview path")
    require(
        preview["source_file"] == "src/nfl_trajectory/portfolio_demo.py",
        "declared demo preview generator",
    )
    require(
        preview["synthetic_only"] is True and preview["seed"] == 2026,
        "preview must use the declared synthetic default seed",
    )
    path = public_path(root, preview["path"])
    source = public_path(root, preview["source_file"])
    require(digest(path) == preview["sha256"], "demo preview digest differs")
    require(digest(source) == preview["source_sha256"], "demo preview source digest differs")
    validate_svg(path)
    labels = " ".join(ElementTree.fromstring(path.read_text()).itertext())
    require("synthetic" in labels.lower(), "demo preview must visibly identify synthetic data")
    if demo_output is not None:
        manifest = json.loads((demo_output / "manifest.json").read_text())
        require(
            manifest["seed"] == preview["seed"]
            and manifest["implementation_sha256"] == preview["source_sha256"],
            "generated demo and preview source/seed differ",
        )
        require(
            path.read_bytes() == (demo_output / "trajectories.svg").read_bytes(),
            "published preview differs from regenerated demo",
        )
    return {
        "synthetic_preview_hash_verified": True,
        "synthetic_preview_replay_verified": demo_output is not None,
    }


def validate(root: Path = ROOT, demo_output: Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "PASS",
        "scope": "curated public closeout release",
        "training_fits": 0,
        "submissions": 0,
    }
    result.update(validate_evidence(root))
    result.update(validate_files(root))
    if demo_output is not None:
        result.update(validate_demo(demo_output))
    result.update(validate_preview(root, demo_output))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-output", type=Path, help="Verify an already generated synthetic demo"
    )
    args = parser.parse_args()
    print(json.dumps(validate(demo_output=args.demo_output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
