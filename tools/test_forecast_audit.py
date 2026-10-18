"""Independent metric, input-boundary and real exported-CSV audit checks."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("forecast_audit", HERE / "audit_public_forecasts.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot import auditor")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class AuditTests(unittest.TestCase):
    def test_two_coordinate_denominator(self) -> None:
        result = AUDIT.pooled([25, 0])
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["sse_xy"], 25)
        self.assertEqual(result["coordinate_rmse_yards"], 2.5)

    def test_keyed_csv_rejects_invalid_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.csv"
            for body in (
                "1,1,1,1,NaN,0",
                "1,1,1,1,inf,0",
                "0,1,1,1,0,0",
                "1,1,1,121,0,0",
                "1,1,1,1,0,0,extra",
                "1,1,1,1,0,0\n1,1,1,1,0,0",
            ):
                path.write_text("game_id,play_id,nfl_id,frame_id,x,y\n" + body + "\n")
                with self.subTest(body=body), self.assertRaises(ValueError):
                    AUDIT.rows(path)

    def test_csv_order_cannot_change_key_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.csv"
            header = "game_id,play_id,nfl_id,frame_id,x,y\n"
            records = ["1,1,1,1,2,3\n", "1,1,1,2,4,5\n"]
            path.write_text(header + "".join(records))
            expected = AUDIT.rows(path)
            path.write_text(header + "".join(reversed(records)))
            self.assertEqual(AUDIT.rows(path), expected)

    def test_paired_comparison_recomputes_pooled_error_after_removing_a_game(self) -> None:
        # An imbalanced fixture catches averaging game RMSEs or averaging deltas.
        first, second, third = (1, 1, 1, 1), (1, 1, 1, 2), (2, 1, 1, 1)
        reference = {first: 0.0, second: 0.0, third: 18.0}
        candidate = {first: 2.0, second: 2.0, third: 0.0}
        result = AUDIT.paired_comparison(reference, candidate)
        self.assertAlmostEqual(result["pooled"]["reference_rmse_yards"], 3**0.5)
        self.assertAlmostEqual(result["pooled"]["candidate_rmse_yards"], (2 / 3) ** 0.5)
        self.assertEqual(result["games_improved"], 1)
        self.assertEqual(result["games_worsened"], 1)
        self.assertEqual(result["leave_one_game_out"]["2"]["rows"], 2)
        self.assertEqual(result["leave_one_game_out"]["2"]["rmse_delta_yards"], 1)
        self.assertEqual(result["leave_one_game_out"]["1"]["rmse_delta_yards"], -3)
        with self.assertRaises(ValueError):
            AUDIT.paired_comparison(reference, {first: 2.0, second: 2.0})

    def test_trajectory_diagnostics_distinguish_coordinate_and_endpoint_error(self) -> None:
        result = AUDIT.trajectory_slices({(1, 2, 3, 1): 25.0, (1, 2, 3, 2): 0.0})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["coordinate_rmse_yards"], 2.5)
        self.assertEqual(result[0]["ade_yards"], 2.5)
        self.assertEqual(result[0]["fde_yards"], 0)
        self.assertEqual(result[0]["maximum_displacement_yards"], 5)
        for invalid in (-1, float("inf"), float("nan")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                AUDIT.pooled([invalid])

    def test_real_export_slices_and_exact_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(
                [
                    sys.executable,
                    "-S",
                    str(HERE.parent / "scripts/run_portfolio_demo.py"),
                    "--output",
                    str(root / "demo"),
                ],
                check=True,
                capture_output=True,
            )
            command = [
                sys.executable,
                "-S",
                str(HERE / "audit_public_forecasts.py"),
                "--demo-output",
                str(root / "demo"),
                "--output",
                str(root / "audit.json"),
                "--html",
                str(root / "diagnostics.html"),
            ]
            subprocess.run(command, check=True, capture_output=True)
            first = (root / "audit.json").read_bytes()
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(first, (root / "audit.json").read_bytes())
            report = json.loads(first)
            for result in report["models"].values():
                self.assertEqual(result["partitions"]["holdout"]["rows"], 144)
                self.assertEqual(
                    sum(r["rows"] for r in result["holdout_horizon_slices"].values()), 144
                )
                self.assertEqual(set(result["holdout_game_slices"]), {"105", "106"})
                self.assertEqual(len(result["holdout_trajectories"]), 12)
                self.assertEqual(sum(row["rows"] for row in result["holdout_trajectories"]), 144)
            self.assertEqual(len(report["paired_comparisons"]), 1)
            self.assertEqual(report["paired_comparisons"][0]["pooled"]["rows"], 144)
            rendered = (root / "diagnostics.html").read_text()
            self.assertIn("synthetic", rendered)
            self.assertIn("Does either game drive", rendered)
            self.assertNotIn("<script", rendered)
            (root / "audit.json").write_text("different artifact")
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual((root / "audit.json").read_text(), "different artifact")


if __name__ == "__main__":
    unittest.main()
