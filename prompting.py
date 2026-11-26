"""
Regras compartilhadas entre o treino (convert_dataset.py), a inferência
(inference.py, session.py) e a GUI (gui.py). Só biblioteca padrão.

O MOOD é uma ENTRADA do modelo, não uma saída. Quem guarda o estado é o
programa: depois de cada resposta ele aplica o `mood_delta` do `answer` ao
mood (apply_mood_delta), tira o mood antigo do contexto e coloca o novo no FIM
do prompt. O mood é salvo à parte, junto do contexto (session.py), e não nos
datasets de treino.

O prompt:
  - uma mensagem JSON por linha, terminando em "\\n" (a fronteira
    prompt/completion cai entre dois tokens);
  - 'branch' nunca entra;
  - 'tool_call' e 'tool_result' só entram no turno atual (depois do último
    'user'): nos turnos passados ficam só user e answer, para economizar espaço;
  - só UM mood aparece, o atual, e é sempre a última linha.
"""
import json

# (chave, rótulo). As quatro últimas dependem mais do personagem: para outro
# personagem, basta editar esta lista (a GUI e o conversor leem daqui).
MOOD_LEVELS = [
    ("shyness_level", "Timidez"),
    ("comfort_level", "Conforto"),
    ("distraction_level", "Distração"),
    ("energy_level", "Energia"),
    ("patience_level", "Paciência"),
    ("curiosity_level", "Curiosidade"),
    ("affection_level", "Afeto"),
    ("trust_level", "Confiança"),
]
MOOD_KEYS = [key for key, _ in MOOD_LEVELS]
MOOD_MIN, MOOD_MAX = 0, 10

# Vocabulário fechado de emoções (primary_emotion do mood e do mood_delta).
# Cobre o espectro realista, inclusive as negativas: a personagem pode ficar
# irritada, com raiva e até furiosa. Agrupadas por família, da mais leve à mais forte.
EMOTIONS = [
    # neutras / calmas
    "calm", "content", "bored", "tired", "distracted", "confused",
    # positivas
    "happy", "excited", "amused", "curious", "proud", "grateful", "tender", "touched",
    # timidez / vergonha / medo
    "shy", "embarrassed", "flustered", "nervous", "worried", "guarded",
    # tristeza
    "sad", "lonely", "disappointed", "hurt",
    # raiva (escala: annoyed < irritated < angry < furious) e aversão
    "uncomfortable", "annoyed", "irritated", "offended", "angry", "furious",
]

# Estado de partida de uma conversa nova: o temperamento base da Lisa
# (character/lisa.md, seção 5): tímida, distraída e muito paciente; ainda não
# confia nem tem afeto por quem acabou de conhecer.
DEFAULT_MOOD = {
    "role": "mood",
    "primary_emotion": "calm",
    "shyness_level": 7,
    "comfort_level": 3,
    "distraction_level": 9,
    "energy_level": 3,
    "patience_level": 10,
    "curiosity_level": 5,
    "affection_level": 3,
    "trust_level": 3,
}

# Mensagens que o MODELO gera (entram na completion do treino).
GENERATED_ROLES = ("tool_call", "answer")


def _clamp(value):
    return max(MOOD_MIN, min(MOOD_MAX, value))


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def mood_delta_problems(delta):
    """Lista o que há de errado num mood_delta (vazia = válido)."""
    if not isinstance(delta, dict):
        return ["mood_delta is missing or not an object"]
    problems = []
    emotion = delta.get("primary_emotion")
    if not isinstance(emotion, str) or not emotion.strip():
        problems.append("primary_emotion is empty")
    elif emotion not in EMOTIONS:
        problems.append(f"primary_emotion {emotion!r} is not in EMOTIONS")
    for key in MOOD_KEYS:
        value = delta.get(key)
        if not _is_int(value) or not -MOOD_MAX <= value <= MOOD_MAX:
            problems.append(f"{key} must be an integer from -10 to 10")
    return problems


def mood_problems(mood):
    """Lista o que há de errado num mood (vazia = válido)."""
    if not isinstance(mood, dict):
        return ["mood is missing or not an object"]
    problems = []
    emotion = mood.get("primary_emotion")
    if not isinstance(emotion, str) or not emotion.strip():
        problems.append("primary_emotion is empty")
    elif emotion not in EMOTIONS:
        problems.append(f"primary_emotion {emotion!r} is not in EMOTIONS")
    for key in MOOD_KEYS:
        value = mood.get(key)
        if not _is_int(value) or not MOOD_MIN <= value <= MOOD_MAX:
            problems.append(f"{key} must be an integer from 0 to 10")
    return problems


def normalize_mood(mood=None):
    """Mood completo, com as chaves na ordem canônica. Faltou algo: usa o padrão."""
    mood = mood if isinstance(mood, dict) else {}
    out = {"role": "mood"}
    emotion = mood.get("primary_emotion")
    out["primary_emotion"] = emotion if isinstance(emotion, str) and emotion.strip() else DEFAULT_MOOD["primary_emotion"]
    for key in MOOD_KEYS:
        value = mood.get(key)
        out[key] = _clamp(value) if _is_int(value) else DEFAULT_MOOD[key]
    return out


def apply_mood_delta(mood, delta):
    """
    Novo mood = mood + delta, com cada nível limitado a 0..10 e a emoção
    trocada pela que o delta traz. Valores inválidos no delta contam como 0
    (o modelo pode errar; o programa não quebra). Não altera os argumentos.
    """
    base = normalize_mood(mood)
    delta = delta if isinstance(delta, dict) else {}
    out = {"role": "mood"}
    emotion = delta.get("primary_emotion")
    out["primary_emotion"] = emotion if isinstance(emotion, str) and emotion.strip() else base["primary_emotion"]
    for key in MOOD_KEYS:
        change = delta.get(key)
        out[key] = _clamp(base[key] + (change if _is_int(change) else 0))
    return out


def render_prompt(messages, mood=None):
    """
    Monta o prompt de um passo. `messages` é a conversa até aqui (pode ter
    moods e branches: eles são descartados). `mood` é o estado atual; se for
    None, vale o último 'mood' que estiver em `messages`.
    """
    if mood is None:
        mood = next((m for m in reversed(messages) if m.get("role") == "mood"), None)

    last_user = max(
        (i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1
    )
    kept = []
    for i, message in enumerate(messages):
        role = message.get("role")
        if role in ("mood", "branch"):
            continue
        if i < last_user and role in ("tool_call", "tool_result"):
            continue
        kept.append(message)
    if mood is not None:
        kept.append(mood)
    return "\n".join(json.dumps(m, ensure_ascii=False) for m in kept) + "\n"
