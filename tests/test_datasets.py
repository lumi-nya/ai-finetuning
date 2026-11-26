import json
import unittest

from tests.helpers import ROOT, quiet
from convert_dataset import conversation_to_examples
from prompting import EMOTIONS, MOOD_KEYS, apply_mood_delta, mood_delta_problems, mood_problems

FILES = sorted((ROOT / "datasets").glob("*.json"))


def turns(conv):
    """[(mood, answer, branches)] por turno."""
    out, i = [], 0
    while i < len(conv):
        if conv[i]["role"] == "user":
            j = i + 1
            while j < len(conv) and conv[j]["role"] != "user":
                j += 1
            seg = conv[i + 1:j]
            out.append((
                next((m for m in seg if m["role"] == "mood"), None),
                next((m for m in seg if m["role"] == "answer"), None),
                [m for m in seg if m["role"] == "branch"],
            ))
            i = j
        else:
            i += 1
    return out


class Datasets(unittest.TestCase):
    def data(self):
        for path in FILES:
            for n, conv in enumerate(json.loads(path.read_text(encoding="utf-8")), 1):
                yield path.name, n, conv

    def test_every_dataset_converts_without_warnings(self):
        for path in FILES:
            for n, conv in enumerate(json.loads(path.read_text(encoding="utf-8")), 1):
                ex, out = quiet(conversation_to_examples, conv, f"{path.name}#{n}")
                self.assertNotIn("WARN", out, f"{path.name} conversa {n}: {out}")
                self.assertTrue(ex)

    def test_moods_and_deltas_are_valid_with_all_variables(self):
        for name, n, conv in self.data():
            for mood, ans, branches in turns(conv):
                where = f"{name}#{n}"
                self.assertEqual(mood_problems(mood), [], where)
                self.assertEqual(mood_delta_problems(ans["mood_delta"]), [], where)
                self.assertEqual(list(mood)[2:], MOOD_KEYS, where)
                for b in branches:
                    self.assertEqual(mood_problems(b["mood"]), [], where)
                    self.assertEqual(mood_delta_problems(b["answer"]["mood_delta"]), [], where)

    def test_mood_chain_is_consistent_inside_each_conversation(self):
        for name, n, conv in self.data():
            prev = None
            for t, (mood, ans, _) in enumerate(turns(conv), 1):
                if prev:
                    self.assertEqual(mood, apply_mood_delta(*prev), f"{name}#{n} turno {t}")
                prev = (mood, ans["mood_delta"])

    def test_mood_is_never_stored_inside_past_prompts(self):
        for name, n, conv in self.data():
            ex, _ = quiet(conversation_to_examples, conv, "x")
            for e in ex:
                self.assertEqual(e["prompt"].count('"role": "mood"'), 1, f"{name}#{n}")
                self.assertEqual(json.loads(e["prompt"].strip().split("\n")[-1])["role"], "mood")

    def test_moods_dataset_exercises_the_character_variables_with_branches(self):
        data = json.loads((ROOT / "datasets" / "moods.json").read_text(encoding="utf-8"))
        varied = set()
        branches = 0
        for conv in data:
            for mood, _, brs in turns(conv):
                for b in brs:
                    branches += 1
                    diff = {k for k in MOOD_KEYS if b["mood"][k] != mood[k]}
                    self.assertTrue(diff, "ramo com o mesmo humor de entrada")
                    varied |= diff
        self.assertGreaterEqual(branches, 8)
        for key in ("patience_level", "curiosity_level", "affection_level", "trust_level"):
            self.assertIn(key, varied)

    def test_every_system_has_the_character_sheet_right_after_role(self):
        text = (ROOT / "character" / "lisa.txt").read_text(encoding="utf-8").strip()
        for name, n, conv in self.data():
            system = conv[0]
            self.assertEqual(system["role"], "system")
            self.assertEqual(list(system)[:2], ["role", "character"], f"{name}#{n}")
            self.assertEqual(system["character"], text, f"{name}#{n}: rode python apply_character.py")

    def test_emotions_come_from_the_closed_vocabulary(self):
        used = set()
        for name, n, conv in self.data():
            for mood, ans, brs in turns(conv):
                used |= {mood["primary_emotion"], ans["mood_delta"]["primary_emotion"]}
                for b in brs:
                    used |= {b["mood"]["primary_emotion"], b["answer"]["mood_delta"]["primary_emotion"]}
        self.assertLessEqual(used, set(EMOTIONS))
        for emotion in ("annoyed", "irritated", "angry", "furious", "hurt"):
            self.assertIn(emotion, used)

    def test_anger_dataset_escalates_and_cools_down(self):
        data = json.loads((ROOT / "datasets" / "raiva.json").read_text(encoding="utf-8"))
        flows = [[ans["mood_delta"]["primary_emotion"] for _, ans, _ in turns(conv)] for conv in data]
        self.assertIn(["annoyed", "irritated", "angry", "angry"], flows)
        self.assertTrue(any("furious" in f for f in flows))
        self.assertIn(["guarded", "hurt", "calm"], flows)

    def test_no_invention_dataset_covers_unknown_facts(self):
        data = json.loads((ROOT / "datasets" / "nao-inventa.json").read_text(encoding="utf-8"))
        misses = [c for c in data if any(m["role"] == "tool_result" and m["content"].get("found") is False for m in c)]
        hits = [c for c in data if any(m["role"] == "tool_result" and m["content"].get("found") is True for m in c)]
        self.assertGreaterEqual(len(misses), 3)
        self.assertGreaterEqual(len(hits), 3)


if __name__ == "__main__":
    unittest.main()
