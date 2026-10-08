"""Public release contracts; standard library only, no private data or services."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[1] / "scripts/validate_portfolio_release.py"
_SPEC = importlib.util.spec_from_file_location("portfolio_release_validator", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release)


class PortfolioReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.evidence_path = self.root / "docs/results/project_closeout.json"
        self.evidence_path.parent.mkdir(parents=True)
        self.original = json.loads(
            (release.ROOT / "docs/results/project_closeout.json").read_text()
        )
        self.evidence_path.write_text(json.dumps(self.original))
        for row in self.original["provenance"]:
            if row["visibility"] == "public historical snapshot":
                public = self.root / row["evidence"]
                public.parent.mkdir(parents=True, exist_ok=True)
                public.write_bytes((release.ROOT / row["evidence"]).read_bytes())

    def test_actual_curated_evidence_has_distinct_evaluation_scopes(self) -> None:
        result = release.validate_evidence(self.root)
        self.assertEqual(result["evaluation_scopes"], 3)
        self.assertEqual(result["local_rmse_arithmetic_checks"], 2)
        self.assertEqual(result["public_provenance_hash_checks"], 1)
        self.assertIs(result["private_source_contents_verified"], False)

    def test_tampered_private_result_is_rejected(self) -> None:
        for key, value in (
            ("private_rmse", 0.44850),
            ("official_rank_claim", True),
            ("late_submission", False),
            ("private_improvement", 0.01),
        ):
            with self.subTest(key=key):
                evidence = copy.deepcopy(self.original)
                evidence["accepted_system"][key] = value
                self.evidence_path.write_text(json.dumps(evidence))
                with self.assertRaisesRegex(ValueError, "PORTFOLIO_RELEASE"):
                    release.validate_evidence(self.root)

    def test_mixed_or_tampered_evaluation_scopes_fail(self) -> None:
        for mutation in ("population", "sse", "duplicate", "private_population"):
            with self.subTest(mutation=mutation):
                evidence = copy.deepcopy(self.original)
                scopes = evidence["evaluation_scopes"]
                if mutation == "population":
                    scopes[1]["rows"] = scopes[2]["rows"]
                elif mutation == "sse":
                    scopes[1]["sse"] += 10
                elif mutation == "duplicate":
                    scopes[2] = copy.deepcopy(scopes[1])
                else:
                    scopes[0]["rows"] = 561607
                self.evidence_path.write_text(json.dumps(evidence))
                with self.assertRaisesRegex(ValueError, "PORTFOLIO_RELEASE"):
                    release.validate_evidence(self.root)

    def test_historical_progression_is_not_an_unchecked_percentage(self) -> None:
        evidence = copy.deepcopy(self.original)
        evidence["historical_comparison"]["relative_reduction_percent"] = 50
        self.evidence_path.write_text(json.dumps(evidence))
        with self.assertRaisesRegex(ValueError, "historical progression arithmetic"):
            release.validate_evidence(self.root)

    def test_public_source_tampering_is_not_hidden_by_private_provenance(self) -> None:
        (self.root / "docs/results/frontier_submission.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "public provenance hash"):
            release.validate_evidence(self.root)

    def test_extra_provenance_preserves_core_claims_and_unique_bounded_inventory(self) -> None:
        for mutation in ("missing_core", "duplicate", "too_many"):
            with self.subTest(mutation=mutation):
                evidence = copy.deepcopy(self.original)
                receipts = evidence["provenance"]
                if mutation == "missing_core":
                    receipts[0]["claim"] = "unrelated additional observation"
                elif mutation == "duplicate":
                    receipts.append(copy.deepcopy(receipts[0]))
                else:
                    for number in range(21 - len(receipts)):
                        row = copy.deepcopy(receipts[0])
                        row["claim"] = "additional observation " + str(number)
                        receipts.append(row)
                self.evidence_path.write_text(json.dumps(evidence))
                with self.assertRaisesRegex(ValueError, "provenance"):
                    release.validate_evidence(self.root)

    def test_local_svg_and_directory_links_resolve(self) -> None:
        (self.root / "docs/plot.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        page = self.root / "README.md"
        page.write_text(
            "![figure](docs/plot.svg) [folder](docs/) [section](#section) "
            "[external](https://example.com/unknown)"
        )
        result = release.validate_files(self.root, ("README.md", "docs/plot.svg"))
        self.assertEqual(result["local_markdown_links_checked"], 2)
        (self.root / "docs/plot.svg").unlink()
        with self.assertRaisesRegex(ValueError, "broken local Markdown link"):
            release.validate_files(self.root, ("README.md",))

    def test_private_artifact_in_explicit_release_files_is_rejected(self) -> None:
        for name in ("model.pt", "return.pyz", "REMOTE_JOURNAL.json", ".env"):
            with self.subTest(name=name):
                (self.root / name).write_text("private fixture")
                with self.assertRaisesRegex(ValueError, "private artifact"):
                    release.validate_files(self.root, (name,))

    def test_new_text_rejects_credentials_signed_urls_and_private_locations(self) -> None:
        payloads = (
            "gh" + "p_" + "A" * 30,
            "https://example.com/file?" + "X-Amz-" + "Signature=abc",
            "/home/" + "sagemaker-user/" + "private/checkpoint",
        )
        for number, payload in enumerate(payloads):
            with self.subTest(case=number):
                (self.root / "README.md").write_text(payload)
                with self.assertRaisesRegex(ValueError, "credential|signed URL|private owner"):
                    release.validate_files(self.root, ("README.md",))

    def test_boundary_does_not_rescan_unlisted_historical_source(self) -> None:
        (self.root / "README.md").write_text("Closed project; synthetic demonstration.")
        (self.root / "historical.py").write_text("/home/" + "sagemaker-user/" + "old-source")
        result = release.validate_files(self.root, ("README.md",))
        self.assertEqual(result["curated_files_checked"], 1)

    def test_front_page_aspiration_is_rejected(self) -> None:
        (self.root / "README.md").write_text("Our next goal is to beat the top score.")
        with self.assertRaisesRegex(ValueError, "competitive aspiration"):
            release.validate_files(self.root, ("README.md",))

    def test_svg_cannot_embed_active_or_external_content(self) -> None:
        path = self.root / "plot.svg"
        for content in ("<script>run()</script>", '<image href="https://example.com/p.png"/>'):
            with self.subTest(content=content):
                path.write_text('<svg xmlns="http://www.w3.org/2000/svg">' + content + "</svg>")
                with self.assertRaisesRegex(ValueError, "active SVG|external SVG"):
                    release.validate_svg(path)

    def test_demo_is_synthetic_hash_verified_and_rejects_mutation(self) -> None:
        source = release.ROOT / "src/nfl_trajectory/portfolio_demo.py"
        spec = importlib.util.spec_from_file_location("_portfolio_demo_release_test", source)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(module)
        output = self.root / "demo"
        module.run_demo(output)
        self.assertEqual(release.validate_demo(output)["synthetic_demo_files_verified"], 7)
        self.make_preview((output / "trajectories.svg").read_bytes())
        self.assertTrue(
            release.validate_preview(self.root, output)["synthetic_preview_replay_verified"]
        )
        with (output / "trajectories.svg").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "published preview differs"):
            release.validate_preview(self.root, output)
        with (output / "predictions.csv").open("a") as stream:
            stream.write("unexpected mutation\n")
        with self.assertRaisesRegex(ValueError, "demo output hash mismatch"):
            release.validate_demo(output)

    def make_preview(self, content: bytes) -> dict[str, object]:
        path = self.root / "docs/assets/demo-preview.svg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        source = self.root / "src/nfl_trajectory/portfolio_demo.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes((release.ROOT / "src/nfl_trajectory/portfolio_demo.py").read_bytes())
        preview = {
            "path": "docs/assets/demo-preview.svg",
            "sha256": release.digest(path),
            "source_file": "src/nfl_trajectory/portfolio_demo.py",
            "source_sha256": release.digest(source),
            "seed": 2026,
            "synthetic_only": True,
        }
        evidence = copy.deepcopy(self.original)
        evidence["public_demo_preview"] = preview
        self.evidence_path.write_text(json.dumps(evidence))
        return preview

    def test_preview_and_generator_tampering_fail_their_own_pins(self) -> None:
        content = b'<svg xmlns="http://www.w3.org/2000/svg"><text>Synthetic demo</text></svg>'
        for relative, message in (
            ("docs/assets/demo-preview.svg", "preview digest"),
            ("src/nfl_trajectory/portfolio_demo.py", "source digest"),
        ):
            with self.subTest(relative=relative):
                self.make_preview(content)
                with (self.root / relative).open("ab") as stream:
                    stream.write(b"\n")
                with self.assertRaisesRegex(ValueError, message):
                    release.validate_preview(self.root)

    def test_rehashed_preview_still_requires_visible_label_and_safe_svg(self) -> None:
        for body, message in (
            ("<text>Observed forecast</text>", "visibly identify synthetic"),
            ("<text>Synthetic</text><script>run()</script>", "active SVG"),
            ('<text>Synthetic</text><image href="https://example.com/x.svg"/>', "external SVG"),
        ):
            with self.subTest(message=message):
                self.make_preview(
                    ('<svg xmlns="http://www.w3.org/2000/svg">' + body + "</svg>").encode()
                )
                with self.assertRaisesRegex(ValueError, message):
                    release.validate_preview(self.root)


if __name__ == "__main__":
    unittest.main()
