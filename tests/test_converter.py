import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import SYSTEM, answer, mood, quiet
from convert_dataset import convert_dataset, conversation_to_examples
from conversation_format import conversation_to_lines, read_conversation_text
from prompting import MOOD_KEYS, apply_mood_delta


def user(text):
    return {"role": "user", "text": text}


CALL = {"role": "tool_call", "name": "recall_fact", "arguments": {"query": "q"}}
RESULT = {"role": "tool_result", "name": "recall_fact", "content": {"found": True}}


class Examples(unittest.TestCase):
    def convert(self, conv):
        return quiet(conversation_to_examples, conv, "t")

    def test_completion_is_only_the_answer_and_mood_is_last_in_prompt(self):
        ex, _ = self.convert([SYSTEM, user("a"), mood("shy", trust=2), answer("x")])
        self.assertEqual(len(ex), 1)
        prompt = [json.loads(l) for l in ex[0]["prompt"].strip().split("\n")]
        self.assertEqual([m["role"] for m in prompt], ["system", "user", "mood"])
        self.assertEqual(prompt[-1]["trust_level"], 2)
        self.assertEqual(json.loads(ex[0]["completion"])["role"], "answer")
        self.assertNotIn('"mood"', ex[0]["completion"].replace('"mood_delta"', ""))

    def test_second_turn_prompt_has_only_the_current_mood(self):
        m1 = mood("shy", trust=2)
        a1 = answer("x", emotion="touched", trust=3)
        m2 = apply_mood_delta(m1, a1["mood_delta"])
        ex, _ = self.convert([SYSTEM, user("a"), m1, a1, user("b"), m2, answer("y")])
        self.assertEqual(len(ex), 2)
        self.assertEqual(ex[1]["prompt"].count('"role": "mood"'), 1)
        last = json.loads(ex[1]["prompt"].strip().split("\n")[-1])
        self.assertEqual((last["primary_emotion"], last["trust_level"]), ("touched", 5))

    def test_tool_cycle_makes_two_examples_sharing_the_input_mood(self):
        m = mood("curious", curiosity=9)
        ex, warns = self.convert([SYSTEM, user("a"), m, CALL, RESULT, answer("x")])
        self.assertEqual(len(ex), 2)
        self.assertEqual(json.loads(ex[0]["completion"])["role"], "tool_call")
        self.assertEqual(json.loads(ex[1]["completion"])["role"], "answer")
        for e in ex:
            self.assertEqual(json.loads(e["prompt"].strip().split("\n")[-1])["curiosity_level"], 9)
        self.assertIn('"role": "tool_result"', ex[1]["prompt"])
        self.assertNotIn("WARN", warns)

    def test_branch_uses_the_same_user_message_with_its_own_input_mood(self):
        main, other = mood("calm", patience=8), mood("irritated", patience=1)
        conv = [SYSTEM, user("again?"), main, answer("ok"),
                {"role": "branch", "mood": other, "answer": answer("no!", emotion="annoyed", patience=-2)}]
        ex, _ = self.convert(conv)
        self.assertEqual(len(ex), 2)
        prompts = [e["prompt"].strip().split("\n") for e in ex]
        self.assertEqual(prompts[0][:-1], prompts[1][:-1])
        self.assertEqual(json.loads(prompts[1][-1])["patience_level"], 1)
        self.assertEqual(json.loads(ex[1]["completion"])["timeline"][1]["text"], "no!")

    def test_branches_do_not_leak_into_following_turns(self):
        m1 = mood("calm")
        a1 = answer("x", emotion="happy", trust=1)
        conv = [SYSTEM, user("a"), m1, a1,
                {"role": "branch", "mood": mood("angry", trust=0), "answer": answer("zzz-branch")},
                user("b"), apply_mood_delta(m1, a1["mood_delta"]), answer("y")]
        ex, _ = self.convert(conv)
        self.assertNotIn("zzz-branch", ex[-1]["prompt"])

    def test_turn_without_mood_is_skipped_with_warning(self):
        ex, out = self.convert([SYSTEM, user("a"), answer("x")])
        self.assertEqual(ex, [])
        self.assertIn("WARN", out)

    def test_four_level_mood_is_flagged(self):
        old = {"role": "mood", "primary_emotion": "calm", "shyness_level": 3, "comfort_level": 7,
               "distraction_level": 3, "energy_level": 5}
        ex, out = self.convert([SYSTEM, user("a"), old, answer("x")])
        self.assertIn("WARN", out)
        self.assertIn("patience_level", out)

    def test_missing_new_variable_in_delta_warns(self):
        bad = answer("x")
        del bad["mood_delta"]["trust_level"]
        _, out = self.convert([SYSTEM, user("a"), mood(), bad])
        self.assertIn("WARN", out)
        self.assertIn("trust_level", out)


class Files(unittest.TestCase):
    def test_pairs_and_conversations_formats_are_equivalent(self):
        conv = [SYSTEM, user("a"), mood("calm", trust=4), CALL, RESULT, answer("x", trust=1),
                {"role": "branch", "mood": mood("guarded", trust=1), "answer": answer("hm")},
                {"role": "branch", "mood": mood("tender", trust=9), "answer": answer("yay")}]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "d.json"
            src.write_text(json.dumps([conv]), encoding="utf-8")
            quiet(convert_dataset, src, tmp / "p")
            quiet(convert_dataset, src, tmp / "c", "conversations")
            pairs = [json.loads(l) for l in (tmp / "p" / "d.jsonl").read_text(encoding="utf-8").splitlines()]
            files = sorted((tmp / "c" / "d").glob("*.jsonl"))
            self.assertEqual(len(files), 1)
            from_file = read_conversation_text(files[0].read_text(encoding="utf-8"), "f")
            self.assertEqual(from_file, pairs)
            self.assertEqual(len(pairs), 4)


class FormatReader(unittest.TestCase):
    def lines(self, conv):
        text, _ = quiet(lambda: "\n".join(conversation_to_lines(conv, "t")) + "\n")
        return text

    def test_branch_markers_present(self):
        text = self.lines([SYSTEM, user("a"), mood(), answer("x"),
                           {"role": "branch", "mood": mood("sad"), "answer": answer("y")}])
        for marker in ("<|response|>", "<|end|>", "<|branch|>", "<|/branch|>"):
            self.assertIn(marker, text)
        self.assertNotIn("<|alt|>", text)

    def test_unclosed_branch_and_orphan_end_are_errors(self):
        good = self.lines([SYSTEM, user("a"), mood(), answer("x"),
                           {"role": "branch", "mood": mood("sad"), "answer": answer("y")}])
        with self.assertRaises(ValueError):
            read_conversation_text(good.replace("<|/branch|>\n", ""), "f")
        with self.assertRaises(ValueError):
            read_conversation_text("<|end|>\n", "f")
        with self.assertRaises(ValueError):
            read_conversation_text(good.replace("<|response|>", "<|response|>\n<|response|>", 1), "f")


if __name__ == "__main__":
    unittest.main()
