"""
Conversa real com o modelo: o PROGRAMA gerencia o mood.

A cada turno:
  1. o programa adiciona a fala do usuário ao contexto;
  2. o modelo recebe o contexto + o mood atual (no fim do prompt) e gera o
     answer (com tool calls no caminho, se precisar);
  3. o programa aplica o `mood_delta` do answer ao mood (apply_mood_delta). O
     mood antigo não fica no contexto: só o novo vai no fim do próximo prompt.

O mood é salvo À PARTE do contexto (chave própria), mas no mesmo arquivo, e é
carregado de volta como as memórias: ele não vai para os datasets de treino.
"""
import json
import os
from pathlib import Path

from prompting import DEFAULT_MOOD, apply_mood_delta, normalize_mood

VERSION = 1


class Session:
    def __init__(self, system, memory=None, mood=None, messages=None):
        self.messages = list(messages) if messages is not None else [system] + ([memory] if memory else [])
        self.mood = normalize_mood(mood if mood is not None else DEFAULT_MOOD)

    def say(self, model, text, tool_handler=None, **options):
        """
        Um turno. `model` precisa ter respond(context, mood, tool_handler, ...)
        (LisaInference tem). Devolve o answer; se o modelo não gerar um, desfaz
        o turno e levanta ValueError.
        """
        size = len(self.messages)
        self.messages.append({"role": "user", "text": text})
        try:
            produced = model.respond(self.messages, self.mood, tool_handler, **options)
            answer = next((m for m in reversed(produced) if m.get("role") == "answer"), None)
            if answer is None:
                raise ValueError("The model did not generate an answer.")
        except Exception:
            del self.messages[size:]
            raise
        self.messages.extend(produced)
        self.mood = apply_mood_delta(self.mood, answer.get("mood_delta"))
        return answer

    def to_dict(self):
        # O mood fica fora da lista de mensagens: é salvo e carregado à parte.
        return {"version": VERSION, "context": self.messages, "mood": self.mood}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or not isinstance(data.get("context"), list):
            raise ValueError("Invalid session file: missing 'context' list.")
        if any(not isinstance(m, dict) or "role" not in m for m in data["context"]):
            raise ValueError("Invalid session file: every context item needs a 'role'.")
        if any(m["role"] == "mood" for m in data["context"]):
            raise ValueError("Invalid session file: the mood is saved apart from the context, not inside it.")
        return cls(system=None, messages=data["context"], mood=data.get("mood"))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
