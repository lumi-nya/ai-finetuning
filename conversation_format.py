"""
Formato de arquivo "uma conversa por arquivo" (.jsonl com marcadores).

Cada linha é UMA mensagem JSON (system, memory, user, mood, tool_call,
tool_result, answer), na ordem da conversa. Linhas que começam com "<|" são
marcadores que dizem o que o modelo deve gerar (a resposta esperada, onde a
perda vale). O mood é ENTRADA: fica logo depois do user, como prompt.

    {"role": "system", ...}                 <- prompt (contexto)
    {"role": "user", "text": "..."}         <- prompt
    {"role": "mood", ...}                   <- prompt: o estado atual (vai no fim)
    <|response|>                            <- abre a resposta esperada
    {"role": "answer", ...}                 <- o modelo gera isto
    <|end|>                                 <- fecha a resposta esperada
    <|branch|>                              <- (opcional) a MESMA fala com OUTRO mood
    {"role": "mood", ...}                      de entrada: o prompt do ramo...
    <|response|>
    {"role": "answer", ...}                    ...e a resposta esperada dele
    <|end|>
    <|/branch|>

Regras:
  - tudo fora de <|response|> ... <|end|> é prompt;
  - cada região vira um exemplo de treino: prompt = tudo que veio antes dela
    (montado por prompting.render_prompt: um só mood, o atual, no fim; sem
    tool_call/tool_result de turnos passados); completion = as linhas da região;
  - um bloco <|branch|> vem logo depois da região que ele alterna e usa o MESMO
    prompt dela, trocando só o mood; ramos não entram no histórico seguinte (a
    conversa continua pela região principal);
  - um turno com tool calling tem duas regiões: [tool_call] e, depois do
    tool_result, [answer];
  - linhas em branco e linhas que começam com "#" são ignoradas.
"""
import json
from pathlib import Path

from prompting import GENERATED_ROLES, render_prompt

RESPONSE = "<|response|>"
END = "<|end|>"
BRANCH = "<|branch|>"
END_BRANCH = "<|/branch|>"


def _line(message):
    return json.dumps(message, ensure_ascii=False)


def _runs(messages):
    """Agrupa mensagens seguidas em ('run', geradas pelo modelo) e ('plain', externas)."""
    groups = []
    for message in messages:
        kind = "run" if message["role"] in GENERATED_ROLES else "plain"
        if groups and groups[-1][0] == kind:
            groups[-1][1].append(message)
        else:
            groups.append((kind, [message]))
    return groups


def conversation_to_lines(conversation, label="conversation"):
    """Converte uma conversa (lista de mensagens) nas linhas do arquivo."""
    lines = []
    user_positions = [i for i, m in enumerate(conversation) if m["role"] == "user"]
    first = user_positions[0] if user_positions else len(conversation)

    for message in conversation[:first]:
        if message["role"] == "branch":
            print(f"[WARN] {label}: branch before the first 'user' message dropped.")
            continue
        lines.append(_line(message))

    for turn, start in enumerate(user_positions, start=1):
        end = user_positions[turn] if turn < len(user_positions) else len(conversation)
        lines.append(_line(conversation[start]))
        segment = conversation[start + 1:end]
        plain = [m for m in segment if m["role"] != "branch"]
        branches = [m for m in segment if m["role"] == "branch"]
        turn_mood = next((m for m in plain if m["role"] == "mood"), None)
        others = [m for m in plain if m["role"] != "mood"]

        if turn_mood is None:
            # Sem mood não há exemplo (o conversor avisa): as mensagens ficam
            # como contexto, sem marcadores.
            lines.extend(_line(m) for m in others)
            if branches:
                print(f"[WARN] {label}, turn {turn}: branches dropped because the turn has no mood.")
            continue

        lines.append(_line(turn_mood))
        groups = _runs(others)
        final_group = next(
            (g for g, (kind, items) in enumerate(groups)
             if kind == "run" and any(m["role"] == "answer" for m in items)),
            None,
        )
        blocks = []
        for number, branch in enumerate(branches, start=2):
            b_mood, b_answer = branch.get("mood"), branch.get("answer")
            if not isinstance(b_mood, dict) or not isinstance(b_answer, dict):
                print(f"[WARN] {label}, turn {turn}, branch {number} dropped: needs 'mood' and 'answer' objects.")
                continue
            blocks.append([BRANCH, _line(b_mood), RESPONSE, _line(b_answer), END, END_BRANCH])
        if blocks and final_group is None:
            print(f"[WARN] {label}, turn {turn}: branches dropped because the turn has no main answer.")
            blocks = []

        for g, (kind, items) in enumerate(groups):
            if kind == "plain":
                lines.extend(_line(m) for m in items)
                continue
            lines.append(RESPONSE)
            lines.extend(_line(m) for m in items)
            lines.append(END)
            if g == final_group:
                for block in blocks:
                    lines.extend(block)

    return lines


def read_conversation_text(text, source="<text>"):
    """Lê o conteúdo de um arquivo e devolve os exemplos {prompt, completion}."""
    history = []          # mensagens já ditas (sem os ramos)
    examples = []
    region = None         # linhas da resposta esperada aberta
    region_context = None  # histórico no momento em que ela abriu
    last_context = None   # contexto da última região principal (para os ramos)
    branch = None         # {"mood": ..., "regions": n} dentro de <|branch|>

    def fail(number, message):
        raise ValueError(f"{source}:{number}: {message}")

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line == RESPONSE:
            if region is not None:
                fail(number, f"{RESPONSE} inside another response.")
            if branch is not None and branch["mood"] is None:
                fail(number, f"a branch needs a mood line before {RESPONSE}.")
            region = []
            region_context = list(history) if branch is None else None
        elif line == END:
            if region is None:
                fail(number, f"{END} without {RESPONSE}.")
            if not region:
                fail(number, "empty response.")
            completion = "\n".join(_line(m) for m in region)
            if branch is None:
                examples.append({"prompt": render_prompt(region_context), "completion": completion})
                last_context = region_context
                history.extend(region)
            else:
                examples.append({"prompt": render_prompt(last_context, mood=branch["mood"]), "completion": completion})
                branch["regions"] += 1
            region = None
        elif line == BRANCH:
            if region is not None or branch is not None:
                fail(number, f"{BRANCH} is not allowed here.")
            if last_context is None:
                fail(number, f"{BRANCH} must come after a response.")
            branch = {"mood": None, "regions": 0}
        elif line == END_BRANCH:
            if branch is None or region is not None:
                fail(number, f"{END_BRANCH} without an open branch.")
            if not branch["regions"]:
                fail(number, "a branch needs at least one response.")
            branch = None
        elif line.startswith("<|"):
            fail(number, f"unknown marker {line!r}.")
        else:
            try:
                message = json.loads(line)
            except json.JSONDecodeError as error:
                fail(number, f"invalid JSON ({error.msg}).")
            if not isinstance(message, dict) or "role" not in message:
                fail(number, "each line must be a JSON object with 'role'.")
            if region is not None:
                region.append(message)
            elif branch is not None:
                if message["role"] != "mood" or branch["mood"] is not None or branch["regions"]:
                    fail(number, "a branch holds one mood line followed by its response.")
                branch["mood"] = message
            else:
                history.append(message)

    if region is not None:
        raise ValueError(f"{source}: response not closed with {END}.")
    if branch is not None:
        raise ValueError(f"{source}: branch not closed with {END_BRANCH}.")
    return examples


def read_conversation_file(path):
    with open(path, "r", encoding="utf-8") as file:
        return read_conversation_text(file.read(), source=str(path))


def load_conversation_examples(directory):
    """Lê todos os arquivos <directory>/<dataset>/<nnn>.jsonl e devolve os exemplos."""
    files = sorted(Path(directory).glob("*/*.jsonl"))
    examples = []
    for path in files:
        examples.extend(read_conversation_file(path))
    return files, examples
