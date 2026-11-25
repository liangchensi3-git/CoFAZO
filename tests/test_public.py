"""Verify the public names against the method switches in the paper experiments."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train


class PublicTrainingTests(unittest.TestCase):
    def test_frozen_source(self):
        digests = json.loads((ROOT / "scripts" / "runtime_sha256.json").read_text("utf8"))
        for name, expected in digests.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest(), expected)

    def test_method_selection(self):
        expected = {"fzoo": ("fzoo", "fzoo", "none"), "dense": ("stored_m", "full", "none"),
                    "factored": ("stored_m", "rc", "none"), "cofazo": ("all_rc4", "rc", "signed_rc4")}
        for dataset in ("sst2", "trec", "cr", "cola", "rte", "mnli", "subj"):
            for method, switches in expected.items():
                args = train.parser().parse_args(["--model", "model", "--output", "out", "--dataset", dataset,
                                                  "--method", method, "--learning-rate", "0.00001"])
                c = train.configuration(args)
                cmd = train.engine_args(c, Path("model"), Path("out"))
                for key, val in zip(("--r39-arm", "--r22-kind", "--r34-m-compression"), switches):
                    self.assertEqual(cmd[cmd.index(key) + 1], val)
                self.assertEqual(c["steps"], 10000)
                self.assertEqual(c["eval_steps"], 250)
                self.assertEqual(c["neval"], train.TASKS[dataset][1])
                self.assertEqual(cmd[cmd.index("--q") + 1], "8")
                self.assertIn("--load_float16", cmd)

    def test_respect_visible_devices(self):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "4,6"}):
            self.assertEqual(train.select_device(1), "6")
            with self.assertRaises(ValueError):
                train.select_device(2)


if __name__ == "__main__":
    unittest.main()
