import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import SYSTEM, answer, mood
import gui
from prompting import MOOD_KEYS


NEUTRAL = dict(patience=6, trust=6, curiosity=5, affection=5)


def neutral(emotion="calm", **levels):
    return mood(emotion, **{**NEUTRAL, **levels})


class ChatApi(unittest.TestCase):
    def setUp(self):
        manager = gui.ModelManager(mock=True, model_name="m", adapter="a")
        manager.model, manager.state = gui.MockModel(), "ready"
        gui.MODEL = manager
        gui.time.sleep = lambda s: None  # sem espera artificial
        self.ctx = [SYSTEM, {"role": "user", "text": "oi"}]

    def chat(self, m=None):
        body = {"context": self.ctx}
        if m is not None:
            body["mood"] = m
        return gui.chat_api(body)

    def test_output_has_only_answer_and_program_applies_delta(self):
        r = self.chat(neutral(energy=5))
        self.assertIsNone(r.get("mood"))
        self.assertEqual(r["answer"]["role"], "answer")
        self.assertEqual(list(r["answer"]["mood_delta"])[1:], MOOD_KEYS)
        self.assertEqual(r["mood_after"]["energy_level"], 6)
        self.assertEqual(r["stopped_at"], "user")  # texto inventado descartado

    def test_prompt_ends_with_the_input_mood(self):
        r = self.chat(mood("sad", trust=1))
        self.assertEqual(json.loads(r["prompt"].strip().split("\n")[-1])["trust_level"], 1)

    def test_variables_change_the_mock_answer(self):
        impatient = self.chat(neutral("irritated", patience=1))
        distrustful = self.chat(neutral("guarded", trust=2))
        curious = self.chat(neutral("curious", curiosity=9))
        loving = self.chat(neutral("tender", affection=9))
        emotions = {r["answer"]["mood_delta"]["primary_emotion"] for r in (impatient, distrustful, curious, loving)}
        self.assertEqual(len(emotions), 4)
        self.assertEqual(impatient["mood_after"]["patience_level"], 0)

    def test_invalid_mood_is_rejected(self):
        with self.assertRaises(gui.ApiError) as e:
            self.chat({**mood(), "trust_level": 99})
        self.assertEqual(e.exception.status, 400)
        with self.assertRaises(gui.ApiError):
            self.chat("sad")

    def test_missing_mood_uses_default(self):
        self.assertIn("answer", self.chat())

    def test_defaults_expose_the_eight_variables(self):
        d = gui.compute_defaults()
        self.assertEqual([x["key"] for x in d["mood_levels"]], MOOD_KEYS)
        self.assertEqual(d["default_mood"]["role"], "mood")


class SharedSystemApi(unittest.TestCase):
    def setUp(self):
        import shared_system as ss
        self.ss, self.tmp = ss, tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.old = (ss.CHARACTER_PATH, ss.SYSTEM_PATH, gui.DATASETS_DIR)
        ss.CHARACTER_PATH, ss.SYSTEM_PATH, gui.DATASETS_DIR = tmp / "lisa.txt", tmp / "system.json", tmp / "datasets"
        ss.save_shared_system({"character": "Lisa.", "animation_catalog": SYSTEM["animation_catalog"], "tools": SYSTEM["tools"]})
        gui.DATASETS_DIR.mkdir()
        self.conv = [{"role": "user", "text": "hi"}, mood(), answer()]
        (gui.DATASETS_DIR / "d.json").write_text(json.dumps([self.conv]), encoding="utf-8")

    def tearDown(self):
        self.ss.CHARACTER_PATH, self.ss.SYSTEM_PATH, gui.DATASETS_DIR = self.old
        self.tmp.cleanup()

    def test_load_injects_and_save_strips(self):
        loaded = gui.load_dataset_api("d")
        self.assertEqual(loaded["data"][0][0]["role"], "system")
        self.assertEqual(loaded["data"][0][0]["character"], "Lisa.")
        gui.save_dataset_api("d", {"data": loaded["data"], "base_hash": loaded["hash"]})
        on_disk = json.loads((gui.DATASETS_DIR / "d.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk[0][0]["role"], "user")

    def test_edited_system_is_kept_as_an_exception(self):
        loaded = gui.load_dataset_api("d")
        loaded["data"][0][0]["character"] = "other"
        gui.save_dataset_api("d", {"data": loaded["data"], "base_hash": loaded["hash"]})
        on_disk = json.loads((gui.DATASETS_DIR / "d.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk[0][0]["character"], "other")

    def test_defaults_come_from_the_shared_system(self):
        d = gui.compute_defaults()
        self.assertEqual(d["character"], "Lisa.")
        self.assertEqual(d["catalog"], SYSTEM["animation_catalog"])

    def test_system_put_validates_and_persists(self):
        new = {"character": "Nova ficha", "animation_catalog": SYSTEM["animation_catalog"], "tools": []}
        self.assertEqual(gui.system_api_put(new)["character"], "Nova ficha")
        self.assertEqual(self.ss.CHARACTER_PATH.read_text(encoding="utf-8").strip(), "Nova ficha")
        for bad in ({**new, "character": " "}, {**new, "animation_catalog": []}, {**new, "tools": "x"}):
            with self.assertRaises(gui.ApiError):
                gui.system_api_put(bad)


class ChatFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = gui.CHATS_DIR
        gui.CHATS_DIR = Path(self.tmp.name) / "chats"

    def tearDown(self):
        gui.CHATS_DIR = self.old
        self.tmp.cleanup()

    def test_save_list_load_keeps_mood_apart_from_context(self):
        ctx = [SYSTEM, {"role": "user", "text": "oi"}, answer("x")]
        ctx[2]["timeline"][0]["blend_weight"] = 1  # int do JS -> float
        gui.save_chat_api("amiga", {"context": ctx, "mood": mood("touched", trust=9)})
        raw = json.loads((gui.CHATS_DIR / "amiga.json").read_text(encoding="utf-8"))
        self.assertEqual(set(raw), {"version", "context", "mood"})
        self.assertEqual(raw["mood"]["trust_level"], 9)
        self.assertTrue(all(m["role"] != "mood" for m in raw["context"]))
        self.assertIsInstance(raw["context"][2]["timeline"][0]["blend_weight"], float)
        listing = gui.list_chats()
        self.assertEqual((listing[0]["name"], listing[0]["emotion"], listing[0]["messages"]), ("amiga", "touched", 2))
        loaded = gui.load_chat_api("amiga")
        self.assertEqual(loaded["mood"]["trust_level"], 9)
        self.assertEqual(loaded["context"], raw["context"])

    def test_validation(self):
        for body in ({"context": "x"}, {"context": [SYSTEM, mood()]},
                     {"context": [SYSTEM], "mood": {**mood(), "trust_level": 50}}, {"context": [SYSTEM], "mood": 3}):
            with self.assertRaises(gui.ApiError) as e:
                gui.save_chat_api("x", body)
            self.assertEqual(e.exception.status, 400)
        for name in ("../x", "", "a b"):
            with self.assertRaises(gui.ApiError):
                gui.chat_path(name)
        with self.assertRaises(gui.ApiError) as e:
            gui.load_chat_api("nope")
        self.assertEqual(e.exception.status, 404)


if __name__ == "__main__":
    unittest.main()
