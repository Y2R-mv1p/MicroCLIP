import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("micropool_eval", REPO / "tools" / "eval.py")
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


class ReleaseTests(unittest.TestCase):
    def test_paper_scope_and_counts(self):
        config, sets, recipes = EVAL.load_release(REPO, EVAL.DEFAULT_SPLITS)
        self.assertEqual(
            {name: len(rows) for name, rows in sets.items()},
            {
                "iid": 5900,
                "partial_ood": 6400,
                "cell_ood": 5900,
                "structure_ood": 5400,
                "lmc_source_shift": 1158,
                "retrieval1000": 1000,
            },
        )
        self.assertNotIn("ood", config["splits"])
        self.assertNotIn("retrieval100", config["splits"])
        self.assertEqual(len(EVAL.unique_images(sets)), 25739)
        self.assertGreater(len(recipes), 25000)

    def test_caption_bindings(self):
        _, sets, _ = EVAL.load_release(REPO, EVAL.DEFAULT_SPLITS)
        structures, cells = EVAL.check_semantics(REPO, sets)
        self.assertTrue(structures)
        self.assertTrue(cells)

    def test_paths_cannot_escape_output(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                EVAL.safe_join(Path(directory), "../escape")
            with self.assertRaises(ValueError):
                EVAL.safe_join(Path(directory), "/absolute")

    def test_manifest_tampering_fails_closed(self):
        config = json.loads((REPO / "config" / "release.json").read_text())
        self.assertEqual(len(config["splits"]), 6)
        original = config["splits"]["iid"]["manifest_sha256"]
        self.assertEqual(len(original), 64)

    def test_public_output_paths(self):
        config = json.loads((REPO / "config" / "release.json").read_text())
        outputs = {item["output"] for item in config["splits"].values()}
        self.assertEqual(
            outputs,
            {
                "evaluation/iid/shard-000000.tar",
                "evaluation/compositional_shift/shard-000000.tar",
                "evaluation/cell_line_shift/shard-000000.tar",
                "evaluation/organelle_set_shift/shard-000000.tar",
                "evaluation/source_shift/shard-000000.tar",
                "evaluation/retrieval_1000/shard-000000.tar",
            },
        )


if __name__ == "__main__":
    unittest.main()
