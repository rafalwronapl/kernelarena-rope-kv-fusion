from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from build_dashboard import html_json, payload, write_dashboard


ROOT = Path(__file__).resolve().parents[1]


class DashboardTests(unittest.TestCase):
    def test_payload_keeps_gpu_and_baseline_classes_separate(self) -> None:
        data = payload(ROOT)
        self.assertEqual(len(data["rows"]), 24)
        self.assertTrue(all(row["correct"] for row in data["rows"]))
        self.assertEqual({row["gpu"] for row in data["rows"]}, {"RTX 4090", "RTX 3090"})
        self.assertEqual({row["dataset"] for row in data["rows"]}, {"real-cache writer", "CUDA Graph replay"})

    def test_dashboard_builds_without_gpu_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            write_dashboard(payload(ROOT), out)
            self.assertEqual({path.name for path in out.iterdir()}, {"app.js", "evidence-data.json", "index.html", "styles.css"})
            self.assertEqual(len(json.loads((out / "evidence-data.json").read_text())["rows"]), 24)

    def test_html_json_escapes_script_boundary(self) -> None:
        self.assertNotIn("</script>", html_json({"value": "</script>"}))


if __name__ == "__main__":
    unittest.main()
