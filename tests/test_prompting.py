import json
import unittest

from tests.helpers import SYSTEM, answer, mood
from prompting import (
    DEFAULT_MOOD, EMOTIONS, MOOD_KEYS, apply_mood_delta, mood_delta_problems, mood_problems,
    normalize_mood, render_prompt,
)


class MoodVariables(unittest.TestCase):
    def test_eight_variables_with_character_dependent_ones(self):
        self.assertEqual(len(MOOD_KEYS), 8)
        for key in ("patience_level", "curiosity_level", "affection_level", "trust_level"):
            self.assertIn(key, MOOD_KEYS)
        self.assertEqual(list(DEFAULT_MOOD), ["role", "primary_emotion"] + MOOD_KEYS)

    def test_problems(self):
        self.assertEqual(mood_problems(mood()), [])
        self.assertTrue(mood_problems({**mood(), "trust_level": 11}))
        self.assertTrue(mood_problems({**mood(), "trust_level": True}))
        self.assertTrue(mood_problems({**mood(), "primary_emotion": " "}))
        broken = mood()
        del broken["patience_level"]
        self.assertTrue(mood_problems(broken))
        self.assertEqual(mood_delta_problems(answer()["mood_delta"]), [])
        self.assertTrue(mood_delta_problems({**answer()["mood_delta"], "trust_level": 11}))
        self.assertTrue(mood_delta_problems(None))

    def test_emotion_vocabulary_is_closed_and_has_the_anger_scale(self):
        for emotion in ("annoyed", "irritated", "angry", "furious", "hurt", "calm"):
            self.assertIn(emotion, EMOTIONS)
        self.assertEqual(len(EMOTIONS), len(set(EMOTIONS)))
        self.assertTrue(mood_problems(mood("hesitant_forgiving")))
        self.assertTrue(mood_delta_problems({**answer()["mood_delta"], "primary_emotion": "smug"}))
        self.assertIn(DEFAULT_MOOD["primary_emotion"], EMOTIONS)

    def test_default_mood_is_lisa_temperament(self):
        self.assertEqual(
            [DEFAULT_MOOD[k] for k in MOOD_KEYS],
            [7, 3, 9, 3, 10, 5, 3, 3],  # timidez, conforto, distração, energia, paciência, curiosidade, afeto, confiança
        )

    def test_normalize_orders_keys_and_fills_defaults(self):
        out = normalize_mood({"trust_level": 9, "primary_emotion": "shy", "role": "mood"})
        self.assertEqual(list(out), ["role", "primary_emotion"] + MOOD_KEYS)
        self.assertEqual(out["trust_level"], 9)
        self.assertEqual(out["patience_level"], DEFAULT_MOOD["patience_level"])
        self.assertEqual(normalize_mood(None), DEFAULT_MOOD)


class ApplyDelta(unittest.TestCase):
    def test_adds_and_replaces_emotion(self):
        out = apply_mood_delta(mood(trust=5), answer(emotion="touched", trust=2, affection=1)["mood_delta"])
        self.assertEqual(out["trust_level"], 7)
        self.assertEqual(out["affection_level"], DEFAULT_MOOD["affection_level"] + 1)
        self.assertEqual(out["primary_emotion"], "touched")

    def test_clamps_to_0_10(self):
        out = apply_mood_delta(mood(trust=9, patience=1), answer(trust=5, patience=-5)["mood_delta"])
        self.assertEqual((out["trust_level"], out["patience_level"]), (10, 0))

    def test_invalid_values_count_as_zero_and_inputs_are_not_mutated(self):
        base = mood(curiosity=4)
        before = json.dumps(base)
        delta = {"primary_emotion": "", "curiosity_level": "3", "trust_level": 2.5, "energy_level": True}
        out = apply_mood_delta(base, delta)
        self.assertEqual(out["curiosity_level"], 4)
        self.assertEqual(out["trust_level"], DEFAULT_MOOD["trust_level"])
        self.assertEqual(out["energy_level"], DEFAULT_MOOD["energy_level"])
        self.assertEqual(out["primary_emotion"], "calm")
        self.assertEqual(json.dumps(base), before)
        self.assertEqual(apply_mood_delta(base, None)["curiosity_level"], 4)


class RenderPrompt(unittest.TestCase):
    def lines(self, text):
        return [json.loads(x) for x in text.strip().split("\n")]

    def test_mood_is_the_last_line_and_the_only_one(self):
        ctx = [SYSTEM, {"role": "user", "text": "a"}, mood("sad"), answer("x"), {"role": "user", "text": "b"}]
        lines = self.lines(render_prompt(ctx, mood=mood("happy", trust=9)))
        self.assertEqual([m["role"] for m in lines], ["system", "user", "answer", "user", "mood"])
        self.assertEqual(lines[-1]["primary_emotion"], "happy")
        self.assertEqual(lines[-1]["trust_level"], 9)

    def test_old_mood_is_removed_and_new_one_goes_last(self):
        ctx = [SYSTEM, {"role": "user", "text": "a"}, mood("sad"), answer("x"), {"role": "user", "text": "b"}]
        new = apply_mood_delta(mood("sad"), answer(emotion="amused", trust=1)["mood_delta"])
        text = render_prompt(ctx, mood=new)
        self.assertEqual(text.count('"role": "mood"'), 1)
        self.assertNotIn('"sad"', text)
        self.assertTrue(text.endswith(json.dumps(new, ensure_ascii=False) + "\n"))

    def test_drops_branches_and_past_tool_messages_keeps_current_ones(self):
        call = {"role": "tool_call", "name": "recall_fact", "arguments": {"query": "q"}}
        result = {"role": "tool_result", "name": "recall_fact", "content": {"found": True}}
        ctx = [SYSTEM, {"role": "user", "text": "a"}, mood(), call, result, answer("x"),
               {"role": "branch", "mood": mood("angry"), "answer": answer("y")},
               {"role": "user", "text": "b"}, call, result]
        roles = [m["role"] for m in self.lines(render_prompt(ctx, mood=mood()))]
        self.assertEqual(roles, ["system", "user", "answer", "user", "tool_call", "tool_result", "mood"])

    def test_without_explicit_mood_uses_the_last_one_in_messages(self):
        ctx = [SYSTEM, {"role": "user", "text": "a"}, mood("sad"), mood("shy")]
        self.assertEqual(self.lines(render_prompt(ctx))[-1]["primary_emotion"], "shy")
        self.assertNotIn('"mood"', render_prompt([SYSTEM, {"role": "user", "text": "a"}]))


if __name__ == "__main__":
    unittest.main()
