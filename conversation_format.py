"""
Formato de arquivo "uma conversa por arquivo" (.jsonl com marcadores).

Cada linha é UMA mensagem JSON (system, memory, user, tool_call, tool_result,
mood, answer), na ordem da conversa. Linhas que começam com "<|" são marcadores
que dizem o que o modelo deve gerar (a resposta esperada, onde a perda vale):

    {"role": "system", ...}                 <- prompt (contexto)
    {"role": "user", "text": "..."}         <- prompt
    <|response|>                            <- abre a resposta esperada
    {"role": "mood", ...}                   <- o modelo gera isto...
    {"role": "answer", ...}                 <- ...e isto
    <|alt|>                                 <- (opcional) outra resposta possível
    {"role": "mood", ...}                      para o MESMO prompt (um "ramo")
    {"role": "answer", ...}
    <|end|>                                 <- fecha a resposta esperada

Regras:
  - tudo fora de <|response|> ... <|end|> é prompt;
  - cada região vira um exemplo de treino: prompt = tudo que veio antes dela
    (montado por prompting.render_prompt: sem mood, e sem tool_call/tool_result
    de turnos passados); completion = as linhas da região;
  - com <|alt|>, cada alternativa vira um exemplo com o mesmo prompt; só a
    PRIMEIRA entra no histórico das regiões seguintes (a conversa continua por
    ela);
  - um turno com tool calling tem duas regiões: [tool_call] e, depois do
    tool_result, [mood + answer];
  - linhas em branco e linhas que começam com "#" são ignoradas.
"""
import json

from prompting import GENERATED_ROLES, render_prompt

RESPONSE = "<|response|>"
ALT = "<|alt|>"
END = "<|end|>"


def _line(message):
    return json.dumps(message, ensure_ascii=False)


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

        # Os ramos viram alternativas da primeira região que tem mood/answer.
        final_group = next(
            (g for g, run in enumerate(_runs(plain))
             if run[0] == "run" and any(m["role"] in ("mood", "answer") for m in run[1])),
            None,
        )
        alternatives = []
        for number, branch in enumerate(branches, start=2):
            mood, answer = branch.get("mood"), branch.get("answer")
            if not isinstance(mood, dict) or not isinstance(answer, dict):
                print(f"[WARN] {label}, turn {turn}, branch {number} dropped: needs 'mood' and 'answer' objects.")
                continue
            alternatives.append([mood, answer])
        if alternatives and final_group is None:
            print(f"[WARN] {label}, turn {turn}: branches dropped because the turn has no main mood/answer.")
            alternatives = []

        for g, (kind, items) in enumerate(_runs(plain)):
            if kind == "plain":
                lines.extend(_line(m) for m in items)
                continue
            lines.append(RESPONSE)
            lines.extend(_line(m) for m in items)
            if g == final_group:
                for alt in alternatives:
                    lines.append(ALT)
                    lines.extend(_line(m) for m in alt)
            lines.append(END)

    return lines


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


def read_conversation_text(text, source="<text>"):
    """Lê o conteúdo de um arquivo e devolve os exemplos {prompt, completion}."""
    history = []        # mensagens já ditas, sem as alternativas
    examples = []
    alts = None         # None = fora de uma resposta esperada

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line == RESPONSE:
            if alts is not None:
                raise ValueError(f"{source}:{number}: {RESPONSE} inside another response.")
            alts = [[]]
        elif line == ALT:
            if alts is None:
                raise ValueError(f"{source}:{number}: {ALT} outside a response.")
            if not alts[-1]:
                raise ValueError(f"{source}:{number}: empty alternative before {ALT}.")
            alts.append([])
        elif line == END:
            if alts is None:
                raise ValueError(f"{source}:{number}: {END} without {RESPONSE}.")
            if any(not alt for alt in alts):
                raise ValueError(f"{source}:{number}: empty response.")
            prompt = render_prompt(history)
            for alt in alts:
                examples.append({"prompt": prompt, "completion": "\n".join(_line(m) for m in alt)})
            history.extend(alts[0])
            alts = None
        elif line.startswith("<|"):
            raise ValueError(f"{source}:{number}: unknown marker {line!r}.")
        else:
            try:
                message = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{number}: invalid JSON ({error.msg}).") from error
            if not isinstance(message, dict) or "role" not in message:
                raise ValueError(f"{source}:{number}: each line must be a JSON object with 'role'.")
            (history if alts is None else alts[-1]).append(message)

    if alts is not None:
        raise ValueError(f"{source}: response not closed with {END}.")
    return examples


def read_conversation_file(path):
    with open(path, "r", encoding="utf-8") as file:
        return read_conversation_text(file.read(), source=str(path))


def load_conversation_examples(directory):
    """Lê todos os arquivos <directory>/<dataset>/<nnn>.jsonl e devolve os exemplos."""
    from pathlib import Path

    files = sorted(Path(directory).glob("*/*.jsonl"))
    examples = []
    for path in files:
        examples.extend(read_conversation_file(path))
    return files, examples
