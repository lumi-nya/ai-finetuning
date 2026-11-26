import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import SYSTEM, answer, mood
from prompting import DEFAULT_MOOD, MOOD_KEYS, render_prompt
from session import Session


class FakeModel:
    def __init__(self, produced):
        self.produced, self.calls = produced, []

    def respond(self, context, mood, tool_handler=None, **options):
        self.calls.append((json.loads(json.dumps(context)), dict(mood), render_prompt(context, mood=mood)))
        return self.produced


class SessionTests(unittest.TestCase):
    def test_new_session_starts_with_default_mood(self):
        self.assertEqual(Session(SYSTEM).mood, DEFAULT_MOOD)

    def test_say_applies_delta_and_next_prompt_has_only_the_new_mood_last(self):
        s = Session(SYSTEM, mood=mood("calm", trust=4))
        model = FakeModel([answer("a", emotion="touched", trust=3, curiosity=1)])
        s.say(model, "hello")
        self.assertEqual((s.mood["trust_level"], s.mood["primary_emotion"]), (7, "touched"))
        self.assertTrue(all(m["role"] != "mood" for m in s.messages))
        s.say(model, "again")
        prompt = model.calls[1][2].strip().split("\n")
        last = json.loads(prompt[-1])
        self.assertEqual(last["role"], "mood")
        self.assertEqual(last["trust_level"], 7)
        self.assertEqual(sum('"role": "mood"' in l for l in prompt), 1)
        self.assertEqual(s.mood["trust_level"], 10)

    def test_first_call_receives_the_mood_before_the_delta(self):
        s = Session(SYSTEM, mood=mood("calm", patience=5))
        model = FakeModel([answer(patience=-3)])
        s.say(model, "x")
        self.assertEqual(model.calls[0][1]["patience_level"], 5)
        self.assertEqual(s.mood["patience_level"], 2)

    def test_no_answer_rolls_back_turn_and_keeps_mood(self):
        s = Session(SYSTEM)
        before, mood_before = list(s.messages), dict(s.mood)
        with self.assertRaises(ValueError):
            s.say(FakeModel([{"role": "tool_call", "name": "recall_fact", "arguments": {}}]), "x")
        self.assertEqual(s.messages, before)
        self.assertEqual(s.mood, mood_before)

    def test_tool_messages_are_kept_in_the_context(self):
        call = {"role": "tool_call", "name": "recall_fact", "arguments": {"query": "q"}}
        result = {"role": "tool_result", "name": "recall_fact", "content": {"found": True}}
        s = Session(SYSTEM)
        s.say(FakeModel([call, result, answer()]), "x")
        self.assertEqual([m["role"] for m in s.messages], ["system", "user", "tool_call", "tool_result", "answer"])


class Persistence(unittest.TestCase):
    def test_mood_is_saved_apart_from_context_in_the_same_file_and_loaded_back(self):
        s = Session(SYSTEM, memory={"role": "memory", "summary": "s", "key_facts": {}}, mood=mood("shy", trust=2))
        s.say(FakeModel([answer(emotion="touched", trust=2)]), "hi")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "chat.json"
            s.save(path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(raw), {"version", "context", "mood"})
            self.assertTrue(all(m["role"] != "mood" for m in raw["context"]))
            self.assertEqual(list(raw["mood"]), ["role", "primary_emotion"] + MOOD_KEYS)
            self.assertEqual(raw["mood"]["trust_level"], 4)
            self.assertEqual([p.name for p in path.parent.iterdir()], ["chat.json"])
            loaded = Session.load(path)
        self.assertEqual(loaded.messages, s.messages)
        self.assertEqual(loaded.mood, s.mood)

    def test_load_rejects_bad_files(self):
        for data in ({}, {"context": "x"}, {"context": [{"text": "no role"}]},
                     {"context": [{"role": "mood", **DEFAULT_MOOD}]}):
            with self.assertRaises(ValueError):
                Session.from_dict(data)

    def test_missing_or_partial_mood_falls_back_to_defaults(self):
        self.assertEqual(Session.from_dict({"context": [SYSTEM]}).mood, DEFAULT_MOOD)
        partial = Session.from_dict({"context": [SYSTEM], "mood": {"trust_level": 9}})
        self.assertEqual(partial.mood["trust_level"], 9)
        self.assertEqual(partial.mood["patience_level"], DEFAULT_MOOD["patience_level"])


if __name__ == "__main__":
    unittest.main()
