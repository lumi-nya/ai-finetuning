import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import SYSTEM, answer, mood, quiet
import shared_system as ss
from convert_dataset import convert_dataset


class SharedSystem(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.old = (ss.CHARACTER_PATH, ss.SYSTEM_PATH)
        ss.CHARACTER_PATH, ss.SYSTEM_PATH = tmp / "c" / "lisa.txt", tmp / "c" / "system.json"
        ss.save_shared_system({"character": "Lisa, test.", "animation_catalog": SYSTEM["animation_catalog"], "tools": SYSTEM["tools"]})
        self.shared = ss.load_shared_system()

    def tearDown(self):
        ss.CHARACTER_PATH, ss.SYSTEM_PATH = self.old
        self.tmp.cleanup()

    def test_build_order_and_roundtrip_on_disk(self):
        self.assertEqual(list(self.shared), ["role", "character", "animation_catalog", "tools"])
        self.assertEqual(self.shared["character"], "Lisa, test.")
        self.assertEqual(sorted(p.name for p in ss.SYSTEM_PATH.parent.iterdir()), ["lisa.txt", "system.json"])
        self.assertEqual(set(json.loads(ss.SYSTEM_PATH.read_text(encoding="utf-8"))), {"animation_catalog", "tools"})

    def test_missing_or_bad_files_raise(self):
        ss.SYSTEM_PATH.write_text('{"tools": []}', encoding="utf-8")
        with self.assertRaises(ValueError):
            ss.load_shared_system()
        ss.SYSTEM_PATH.unlink()
        with self.assertRaises(OSError):
            ss.load_shared_system()

    def test_with_and_without_are_inverse_and_respect_overrides(self):
        conv = [{"role": "user", "text": "a"}, mood(), answer()]
        full = ss.with_shared_system(conv, self.shared)
        self.assertEqual(full[0], self.shared)
        self.assertIsNot(full[0], self.shared)  # cópia
        self.assertEqual(ss.without_shared_system(full, self.shared), conv)
        own = [{**self.shared, "character": "outra"}] + conv
        self.assertIs(ss.with_shared_system(own, self.shared), own)
        self.assertEqual(ss.without_shared_system(own, self.shared), own)  # exceção continua no dataset

    def test_converter_joins_the_system_when_generating_training_data(self):
        conv = [{"role": "user", "text": "hi"}, mood(), answer("x")]
        with tempfile.TemporaryDirectory() as out:
            src = Path(out) / "d.json"
            src.write_text(json.dumps([conv]), encoding="utf-8")
            quiet(convert_dataset, src, Path(out) / "p")
            quiet(convert_dataset, src, Path(out) / "c", "conversations")
            line = json.loads((Path(out) / "p" / "d.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(json.loads(line["prompt"].split("\n", 1)[0]), self.shared)
            file = next((Path(out) / "c" / "d").glob("*.jsonl"))
            self.assertIn('"character": "Lisa, test."', file.read_text(encoding="utf-8"))
            self.assertNotIn("character", src.read_text(encoding="utf-8"))

    def test_conversation_with_own_system_keeps_it(self):
        own = {**SYSTEM, "character": "exceção"}
        with tempfile.TemporaryDirectory() as out:
            src = Path(out) / "d.json"
            src.write_text(json.dumps([[own, {"role": "user", "text": "hi"}, mood(), answer()]]), encoding="utf-8")
            quiet(convert_dataset, src, Path(out) / "p")
            line = json.loads((Path(out) / "p" / "d.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(json.loads(line["prompt"].split("\n", 1)[0])["character"], "exceção")


if __name__ == "__main__":
    unittest.main()
