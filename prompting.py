"""
Monta o prompt exatamente do mesmo jeito no treino (convert_dataset.py), na
inferência (inference.py) e na GUI (gui.py). Só biblioteca padrão.

Regras:
  - uma mensagem JSON por linha, terminando em "\\n" (a fronteira
    prompt/completion cai entre dois tokens);
  - 'mood' e 'branch' nunca entram (o modelo é quem gera o mood do turno);
  - 'tool_call' e 'tool_result' só entram no turno atual (depois do último
    'user'): nos turnos passados ficam só user e answer, para economizar espaço.
"""
import json

# Mensagens que o MODELO gera (entram na completion do treino).
GENERATED_ROLES = ("tool_call", "mood", "answer")


def render_prompt(messages):
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
    return "\n".join(json.dumps(m, ensure_ascii=False) for m in kept) + "\n"
