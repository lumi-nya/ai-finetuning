import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompting import DEFAULT_MOOD, MOOD_KEYS  # noqa: E402

SYSTEM = {
    "role": "system",
    "animation_catalog": [{"id": "wave", "description": "d", "category": "gesture", "layer": "body", "tags": []}],
    "tools": [{"name": "recall_fact", "description": "d", "parameters": {"query": "string"}}],
}


def mood(emotion="calm", **levels):
    out = {"role": "mood", "primary_emotion": emotion}
    for key in MOOD_KEYS:
        out[key] = levels.get(key.replace("_level", ""), DEFAULT_MOOD[key])
    return out


def answer(text="hi", emotion="happy", **delta):
    d = {"primary_emotion": emotion}
    for key in MOOD_KEYS:
        d[key] = delta.get(key.replace("_level", ""), 0)
    return {
        "role": "answer", "agent_id": "Lisa",
        "timeline": [{"event_role": "action", "animation_id": "wave", "blend_weight": 1.0},
                     {"event_role": "speech", "text": text}],
        "mood_delta": d,
    }


def quiet(function, *args, **kwargs):
    """Roda e devolve (resultado, texto impresso)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*args, **kwargs)
    return result, buffer.getvalue()
