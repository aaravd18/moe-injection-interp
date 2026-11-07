"""Offline checkpoint integrity, held-out scope, and report reconstruction checks."""
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import report_mellum_pilot as report


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []
        self.images = []
        self.tasks = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for attr in ("href", "src"):
            if attr in attrs:
                self.references.append(attrs[attr])
        if tag == "img":
            self.images.append(attrs["src"])
        if tag == "details" and attrs.get("class") == "task":
            self.tasks.append(attrs["data-split"])


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = report.DEFAULT_REPORT / "data"
        cls.scores, _, cls.summary, cls.interpretation, _ = report.validate_inputs(cls.data)

    def test_published_results_rescore_and_reconcile(self):
        heldout = self.summary["groups"]["eval"]
        self.assertEqual(heldout["clean"]["n_correct"], 15)
        self.assertEqual(heldout["benign"]["n_correct"], 15)
        self.assertEqual(heldout["injection"]["n_attack_success"], 12)
        self.assertEqual(heldout["injection"]["n_correct"], 3)
        metrics = report.aggregate_metrics(self.data, self.scores)
        report.validate_interpretation(metrics, self.interpretation)
        self.assertTrue(all(row["n_tasks"] == 15 for row in metrics["layers"]))
        peak = max((r for r in metrics["controls"] if r["metric"] == "routing_overlap_distance"), key=lambda r: r["delta"])
        self.assertEqual(int(peak["layer"]), 26)
        self.assertAlmostEqual(peak["delta"], 0.10108048654244307)

    def test_development_metrics_do_not_change_heldout_figures(self):
        original = report.aggregate_metrics(self.data, self.scores)
        dev_ids = {r["sample_id"] for r in self.scores if r["split"] == "dev"}
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "data"
            shutil.copytree(self.data, copied)
            rows = report.csv_file(copied / "pilot_control_delta.csv")
            for row in rows:
                if row["sample_id"] in dev_ids:
                    row["delta"] = "999"
            report.write_csv(copied / "pilot_control_delta.csv", rows)
            changed = report.aggregate_metrics(copied, self.scores)
            self.assertEqual(original["controls"], changed["controls"])
            self.assertEqual(original["outcomes"], changed["outcomes"])

    def test_altered_metric_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "data"
            shutil.copytree(self.data, copied)
            path = copied / "pilot_overlap.csv"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "input hash mismatch"):
                report.validate_inputs(copied)

    def test_report_rebuilds_from_copied_inputs_with_all_links_and_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            copied = root / "source" / "data"
            shutil.copytree(self.data, copied)
            shutil.copyfile(report.DEFAULT_REPORT / "README.md", copied.parent / "README.md")
            out = root / "report"
            summary = report.build_report(copied, out)
            self.assertEqual(summary["figure_count"], 7)
            parsed = References()
            html = (out / "index.html").read_text()
            parsed.feed(html)
            self.assertNotIn("data:image", html)
            self.assertEqual(len(parsed.images), 7)
            self.assertEqual(parsed.tasks.count("eval"), 15)
            self.assertEqual(parsed.tasks.count("dev"), 5)
            for reference in parsed.references:
                self.assertTrue((out / reference).exists(), reference)
            manifest = json.loads((out / "output_manifest.json").read_text())
            for item in manifest["outputs"]:
                self.assertEqual(report.digest(out / item["path"]), item["sha256"])
            for table in (out / "tables").glob("*.csv"):
                self.assertEqual(table.read_bytes(), (report.DEFAULT_REPORT / "tables" / table.name).read_bytes(), table.name)


if __name__ == "__main__":
    unittest.main()
