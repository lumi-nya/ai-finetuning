import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import SYSTEM
import apply_character


class ApplyCharacter(unittest.TestCase):
    def test_inserts_after_role_updates_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.json"
            path.write_text(json.dumps([[SYSTEM, {"role": "user", "text": "a"}]], indent=4) + "\n", encoding="utf-8")
            self.assertEqual(apply_character.apply(path, "ficha 1"), 1)
            system = json.loads(path.read_text(encoding="utf-8"))[0][0]
            self.assertEqual(list(system), ["role", "character", "animation_catalog", "tools"])
            self.assertEqual(apply_character.apply(path, "ficha 1"), 0)
            self.assertEqual(apply_character.apply(path, "ficha 2"), 1)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))[0][0]["character"], "ficha 2")
            self.assertTrue(path.read_text(encoding="utf-8").endswith("]\n"))


if __name__ == "__main__":
    unittest.main()
