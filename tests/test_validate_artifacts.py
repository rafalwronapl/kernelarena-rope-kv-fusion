from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from validate_artifacts import manifest_rows, validate, validate_benchmarks


ROOT = Path(__file__).resolve().parents[1]


class ArtifactValidationTests(unittest.TestCase):
    def test_recorded_artifacts_and_manifest_validate(self) -> None:
        summary, errors = validate(ROOT)
        self.assertEqual(errors, [])
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["rtx4090"]["rows"], 16)
        self.assertEqual(summary["rtx3090_graph"]["correct"], 8)

    def test_manifest_has_substantial_coverage(self) -> None:
        self.assertGreaterEqual(len(manifest_rows(ROOT / "ARTIFACT_MANIFEST.md")), 40)

    def test_corrupt_correctness_flag_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            (temp / "artifacts").mkdir()
            for name in (
                "real_vllm_4090_rope_real_vllm_contract_summary.json",
                "real_vllm_cudagraph_decode_summary_3090_postreplay.json",
                "rope_provider_split_3090.json",
            ):
                source = ROOT / "artifacts" / name
                (temp / "artifacts" / name).write_bytes(source.read_bytes())
            path = temp / "artifacts" / "real_vllm_cudagraph_decode_summary_3090_postreplay.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["rows"][0]["graph_correct"] = False
            path.write_text(json.dumps(data), encoding="utf-8")

            _, errors = validate_benchmarks(temp)

            self.assertTrue(any("graph correctness failure" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
