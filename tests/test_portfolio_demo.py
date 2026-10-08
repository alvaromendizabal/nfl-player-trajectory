"""Public demo checks that also run with Python's site packages disabled."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "nfl_trajectory" / "portfolio_demo.py"
SPEC = importlib.util.spec_from_file_location("_tested_portfolio_demo", SOURCE)
assert SPEC is not None and SPEC.loader is not None
DEMO = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DEMO
SPEC.loader.exec_module(DEMO)


def row(frame: int, x: float = 0.0, y: float = 0.0) -> dict[str, int | float]:
    return {"game_id": 101, "play_id": 1, "nfl_id": 1, "frame_id": frame, "x": x, "y": y}


class GuardedRequest(dict[str, object]):
    """Fail if a predictor inspects an attached future target coordinate."""

    def __getitem__(self, key: str) -> object:
        if key in ("x", "y"):
            raise AssertionError("The predictor read a future coordinate")
        return super().__getitem__(key)


class PortfolioDemoTests(unittest.TestCase):
    def test_metric_counts_both_coordinates_and_aligns_keys(self) -> None:
        predictions = [row(1, 3, 4), row(2)]
        labels = [row(2), row(1)]
        self.assertEqual(DEMO.coordinate_rmse(predictions, labels), 2.5)
        self.assertNotEqual(DEMO.coordinate_rmse(predictions, labels), math.sqrt(25 / 2))

    def test_forecast_is_observation_only_and_preserves_request_order(self) -> None:
        observations = [row(2, 1, 2), row(1)]
        requests = [GuardedRequest(row(3, -9999, 9999)), GuardedRequest(row(1, 1234, -1234))]
        self.assertEqual(
            DEMO.predict_observed(observations, requests), [row(3, 4, 8), row(1, 2, 4)]
        )
        self.assertEqual(
            DEMO.predict_observed(observations, requests, "hold_last_position"),
            [row(3, 1, 2), row(1, 1, 2)],
        )

    def test_future_labels_cannot_change_predictions(self) -> None:
        data = DEMO.synthetic_data()
        predictions = DEMO.predict_observed(data.observations, data.requests)
        for label in data.labels:
            label["x"] = 1_000_000.0
            label["y"] = -1_000_000.0
        self.assertEqual(DEMO.predict_observed(data.observations, data.requests), predictions)
        self.assertGreater(DEMO.coordinate_rmse(predictions, data.labels), 900_000)

    def test_invalid_forecast_inputs_fail(self) -> None:
        good = [row(1), row(2, 1, 2)]
        cases = [
            (good + [row(1)], [row(1)], "Duplicate observation"),
            ([row(1), row(3)], [row(1)], "contiguous"),
            ([row(1)], [row(1)], "at least two"),
            ([row(1), row(2, float("nan"))], [row(1)], "finite"),
            ([row(1), row(2, float("inf"))], [row(1)], "finite"),
            (good, [row(1), row(1)], "Duplicate request"),
            (good, [{**row(1), "nfl_id": 2}], "no observed history"),
            (good, [{**row(1), "frame_id": 1.5}], "positive integer"),
            (good, [{**row(1), "game_id": True}], "positive integer"),
            (good, [row(121)], "horizon"),
            (good, [{"game_id": 101}], "Missing identifier"),
        ]
        for observations, requests, expected in cases:
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                DEMO.predict_observed(observations, requests)
        with self.assertRaisesRegex(ValueError, "finite"):
            DEMO.predict_observed([row(1, -1e308), row(2, 1e308)], [row(1)])

    def test_metric_rejects_duplicate_missing_extra_and_nonfinite_rows(self) -> None:
        cases = [
            ([row(1)], [row(1), row(1)]),
            ([row(1), row(1)], [row(1)]),
            ([row(1)], [row(1), row(2)]),
            ([row(1), row(2)], [row(1)]),
            ([row(1, float("nan"))], [row(1)]),
            ([row(1)], [row(1, float("inf"))]),
            ([], []),
        ]
        for predictions, labels in cases:
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                DEMO.coordinate_rmse(predictions, labels)

    def test_game_partitions_and_generation_are_deterministic(self) -> None:
        before = random.getstate()
        first = DEMO.synthetic_data()
        self.assertEqual(random.getstate(), before)
        self.assertEqual(first, DEMO.synthetic_data())
        self.assertNotEqual(first, DEMO.synthetic_data(7))
        groups = [
            {game for game, value in first.splits.items() if value == split}
            for split in ("train", "validation", "holdout")
        ]
        self.assertEqual([len(group) for group in groups], [3, 1, 2])
        self.assertEqual(len(set.union(*groups)), sum(map(len, groups)))
        self.assertEqual(len(first.observations), 288)
        self.assertEqual(len(first.labels), 432)
        self.assertEqual({entry["game_id"] for entry in first.labels}, set.union(*groups))
        self.assertTrue(all(set(request) == set(DEMO.KEYS) for request in first.requests))

    def test_replay_is_byte_identical_and_all_receipts_match(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            left, right = Path(folder) / "left", Path(folder) / "right"
            first, second = DEMO.run_demo(left), DEMO.run_demo(right)
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
            self.assertEqual(DEMO.run_demo(left), first)
            self.assertEqual(len(list(left.iterdir())), 8)
            for path in left.iterdir():
                self.assertEqual(path.read_bytes(), (right / path.name).read_bytes())
            manifest = json.loads((left / "manifest.json").read_text())
            self.assertEqual(
                manifest["implementation_sha256"], hashlib.sha256(SOURCE.read_bytes()).hexdigest()
            )
            self.assertEqual(len(manifest["files"]), 7)
            for receipt in manifest["files"]:
                content = (left / receipt["file"]).read_bytes()
                self.assertEqual(len(content), receipt["bytes"])
                self.assertEqual(hashlib.sha256(content).hexdigest(), receipt["sha256"])
            self.assertNotIn(folder, (left / "manifest.json").read_text())
            self.assertTrue(manifest["synthetic_only"])

    def test_exported_csv_independently_reproduces_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            DEMO.run_demo(output)
            metrics = json.loads((output / "metrics.json").read_text())
            holdout = metrics["partitions"]["holdout"]
            with (output / "labels.csv").open(newline="") as stream:
                labels = {
                    tuple(item[key] for key in DEMO.KEYS): item
                    for item in csv.DictReader(stream)
                    if int(item["game_id"]) in holdout["games"]
                }
            with (output / "predictions.csv").open(newline="") as stream:
                predictions = list(csv.DictReader(stream))
            for method in DEMO.MODELS:
                errors: list[float] = []
                count = 0
                for prediction in predictions:
                    key = tuple(prediction[name] for name in DEMO.KEYS)
                    if prediction["model"] != method or key not in labels:
                        continue
                    count += 1
                    errors.extend(
                        (float(prediction[name]) - float(labels[key][name])) ** 2
                        for name in ("x", "y")
                    )
                self.assertEqual(count, 144)
                actual = math.sqrt(math.fsum(errors) / (2 * count))
                self.assertAlmostEqual(actual, holdout["coordinate_rmse_yards"][method], places=11)
            self.assertEqual(holdout["coordinate_rmse_yards"]["constant_velocity"], 0.263475522816)
            self.assertEqual(metrics["training_fits"], 0)
            self.assertTrue(metrics["synthetic_only"])

    def test_existing_different_output_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "metrics.json").write_text("existing user file")
            with self.assertRaisesRegex(ValueError, "different content"):
                DEMO.run_demo(output)
            self.assertEqual((output / "metrics.json").read_text(), "existing user file")
            self.assertEqual(len(list(output.iterdir())), 1)

    def test_dashboard_and_svg_are_self_contained_and_labelled(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            DEMO.run_demo(output)
            svg = (output / "trajectories.svg").read_text()
            document = ET.fromstring(svg)
            ns = {"svg": "http://www.w3.org/2000/svg"}
            self.assertEqual(len(document.findall("svg:polyline", ns)), 9)
            self.assertIn("Synthetic", document.findtext("svg:title", namespaces=ns) or "")
            html = (output / "dashboard.html").read_text()
            self.assertIn(svg.strip(), html)
            self.assertIn("simulated-data metrics", html)
            self.assertNotIn("<script", html)
            self.assertNotIn("<link", html)
            self.assertNotIn("<iframe", html)

    def test_cli_runs_without_site_packages_from_another_directory(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            command = [
                sys.executable,
                "-S",
                str(ROOT / "scripts" / "run_portfolio_demo.py"),
                "--output",
                "output",
            ]
            completed = subprocess.run(
                command, cwd=folder, capture_output=True, text=True, check=False, timeout=20
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('"status": "PASS"', completed.stdout)
            self.assertTrue((Path(folder) / "output" / "dashboard.html").is_file())
            failed = subprocess.run(
                command + ["--seed", "-1"],
                cwd=folder,
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )
            self.assertEqual(failed.returncode, 1)
            self.assertIn("Seed must be an integer", failed.stderr)


if __name__ == "__main__":
    unittest.main()
