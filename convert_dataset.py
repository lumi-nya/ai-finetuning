
import json
import argparse
import os
from pathlib import Path

from conversation_format import conversation_to_lines, read_conversation_file
from prompting import GENERATED_ROLES, mood_delta_problems, mood_problems, render_prompt

ROOT = Path(__file__).resolve().parent

# Nome usado por versões antigas deste script (um arquivo por conversa, uma
# mensagem por linha). Esses arquivos têm outro formato e precisam sair do
# caminho, senão o train_model.py mistura os dois.
LEGACY_PATTERN = "dataset-*.jsonl"


def to_lines(messages: list[dict]) -> list[str]:
    return [json.dumps(message, ensure_ascii=False) for message in messages]


def conversation_to_examples(conversation: list[dict], label: str) -> list[dict]:
    """
    Um exemplo de treino por PASSO GERADO pelo modelo, no formato
    prompt/completion. O MOOD é entrada: cada turno tem um 'mood' (o estado
    quando o usuário falou) e ele vai no FIM do prompt de todos os passos do
    turno. O modelo só gera tool_call e answer (que termina em mood_delta):

      user + mood -> [tool_call+] -> tool_result+ -> [tool_call+ -> tool_result+] -> answer

    Cada sequência seguida de mensagens do modelo (tool_call, answer) vira uma
    completion; as externas (user, tool_result) só entram no prompt do passo
    seguinte. O prompt é montado por prompting.render_prompt, igual na
    inferência: um só mood (o atual, no fim), sem branch, e sem
    tool_call/tool_result dos turnos passados.

    Um turno pode ter respostas alternativas (mensagens 'branch' com 'mood' e
    'answer'): é a MESMA fala do usuário com OUTRO mood de entrada. Cada uma
    vira um exemplo extra (prompt com o mood do ramo, completion = answer do
    ramo). Os ramos não entram no contexto dos turnos seguintes.

    O treino calcula a perda só na completion.
    """
    user_positions = [
        i for i, message in enumerate(conversation) if message["role"] == "user"
    ]

    if not user_positions:
        print(f"[WARN] {label} ignored: no 'user' message.")
        return []

    system = next((m for m in conversation if m["role"] == "system"), {})
    tool_names = {
        t.get("name") for t in (system.get("tools") or []) if isinstance(t, dict)
    }
    examples = []

    for turn, start in enumerate(user_positions, start=1):
        is_last = turn == len(user_positions)
        end = len(conversation) if is_last else user_positions[turn]
        segment = [i for i in range(start + 1, end) if conversation[i]["role"] != "branch"]
        branches = [
            conversation[i] for i in range(start + 1, end) if conversation[i]["role"] == "branch"
        ]

        if not segment and not branches:
            print(f"[WARN] {label}, turn {turn} ignored: nothing after the user message.")
            continue

        turn_mood = next((conversation[i] for i in segment if conversation[i]["role"] == "mood"), None)
        if turn_mood is None:
            print(f"[WARN] {label}, turn {turn} ignored: no 'mood' (the mood is an input and the prompt needs it).")
            continue

        problems = mood_problems(turn_mood)
        if problems:
            print(f"[WARN] {label}, turn {turn}: mood: {'; '.join(problems)}.")

        positions = [i for i in segment if conversation[i]["role"] != "mood"]

        def prefix(upto):
            return [conversation[i] for i in range(upto) if conversation[i]["role"] != "branch"]

        pending_calls = 0
        final_done = False
        k = 0
        while k < len(positions):
            role = conversation[positions[k]]["role"]
            if role in GENERATED_ROLES:
                run = [positions[k]]
                while k + 1 < len(positions) and conversation[positions[k + 1]]["role"] in GENERATED_ROLES:
                    k += 1
                    run.append(positions[k])
                messages = [conversation[i] for i in run]
                context = prefix(run[0])
                examples.append({
                    "prompt": render_prompt(context, mood=turn_mood),
                    "completion": "\n".join(to_lines(messages)),
                })
                has_answer = False
                for m in messages:
                    if m["role"] == "tool_call":
                        pending_calls += 1
                        if tool_names and m.get("name") not in tool_names:
                            print(f"[WARN] {label}, turn {turn}: tool_call '{m.get('name')}' is not listed in system.tools.")
                        elif not tool_names:
                            print(f"[WARN] {label}, turn {turn}: tool_call but the system message has no 'tools'.")
                    elif m["role"] == "answer":
                        has_answer = True
                        for problem in mood_delta_problems(m.get("mood_delta")):
                            print(f"[WARN] {label}, turn {turn}: answer.mood_delta: {problem}.")
                if pending_calls and has_answer:
                    print(f"[WARN] {label}, turn {turn}: answer generated before the tool_result.")
                if has_answer and not final_done:
                    final_done = True
                    for number, branch in enumerate(branches, start=2):
                        b_mood, b_answer = branch.get("mood"), branch.get("answer")
                        if not isinstance(b_mood, dict) or not isinstance(b_answer, dict):
                            print(f"[WARN] {label}, turn {turn}, branch {number} ignored: needs 'mood' and 'answer' objects.")
                            continue
                        for problem in mood_problems(b_mood):
                            print(f"[WARN] {label}, turn {turn}, branch {number}: mood: {problem}.")
                        for problem in mood_delta_problems(b_answer.get("mood_delta")):
                            print(f"[WARN] {label}, turn {turn}, branch {number}: answer.mood_delta: {problem}.")
                        examples.append({
                            "prompt": render_prompt(context, mood=b_mood),
                            "completion": "\n".join(to_lines([b_answer])),
                        })
            else:
                if role == "tool_result":
                    if pending_calls == 0:
                        print(f"[WARN] {label}, turn {turn}: tool_result without a tool_call before it.")
                    pending_calls = max(0, pending_calls - 1)
            k += 1

        if pending_calls:
            print(f"[WARN] {label}, turn {turn}: tool_call without a tool_result.")

        if branches and not final_done:
            print(f"[WARN] {label}, turn {turn}: branches ignored because the turn has no main answer.")

    return examples


def convert_dataset(input_path: Path, output_dir: Path, fmt: str = "pairs"):
    print(f"Processing: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError(f"{input_path}: the JSON root must be a list.")

    examples = []
    used_conversations = 0

    for index, conversation in enumerate(dataset, start=1):
        if not isinstance(conversation, list):
            print(f"[WARN] Context {index} ignored: not a list.")
            continue

        if not conversation:
            print(f"[WARN] Context {index} ignored: empty.")
            continue

        for message_index, message in enumerate(conversation, start=1):
            if not isinstance(message, dict):
                raise ValueError(
                    f"{input_path}: message {message_index} "
                    f"in context {index} is not an object."
                )

            if "role" not in message:
                raise ValueError(
                    f"{input_path}: message {message_index} "
                    f"in context {index} missing role."
                )

        converted = conversation_to_examples(conversation, f"Context {index}")

        if converted:
            used_conversations += 1
            examples.extend(converted)

    if fmt == "conversations":
        write_conversation_files(dataset, examples, input_path, output_dir)
        print(f"Finished: {input_path}\n")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Um arquivo por dataset de origem: datasets/playful.json -> .datasets/playful.jsonl
    output_path = output_dir / f"{input_path.stem}.jsonl"

    with output_path.open("w", encoding="utf-8") as output:
        for example in examples:
            output.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(
        f"  -> {output_path} "
        f"({len(examples)} examples from {used_conversations} conversations)"
    )
    print(f"Finished: {input_path}\n")


def write_conversation_files(dataset, expected, input_path: Path, output_dir: Path):
    """
    Formato 'conversations': uma conversa por arquivo, uma mensagem por linha,
    com marcadores <|response|> / <|alt|> / <|end|> (ver conversation_format.py).
    Depois de gravar, lê os arquivos de volta e confere que dão exatamente os
    mesmos exemplos do formato 'pairs'.
    """
    target = output_dir / input_path.stem
    target.mkdir(parents=True, exist_ok=True)

    for stale in target.glob("*.jsonl"):
        stale.unlink()

    written = []
    for index, conversation in enumerate(dataset, start=1):
        if not isinstance(conversation, list) or not conversation:
            continue
        if not any(m["role"] == "user" for m in conversation):
            continue
        path = target / f"{index:03d}.jsonl"
        path.write_text("\n".join(conversation_to_lines(conversation, f"Context {index}")) + "\n", encoding="utf-8")
        written.append(path)

    got = [example for path in written for example in read_conversation_file(path)]
    if got != expected:
        raise ValueError(
            f"{input_path}: the written conversation files do not reproduce the training examples "
            f"({len(got)} read back vs {len(expected)} expected)."
        )

    print(f"  -> {target}{os.sep} ({len(written)} files, {len(got)} examples, verified)")


def remove_legacy_outputs(output_dir: Path):
    for stale in sorted(output_dir.glob(LEGACY_PATTERN)):
        stale.unlink()
        print(f"[INFO] Removed old-format file: {stale}")


def main():
    parser = argparse.ArgumentParser(description="Convert JSON datasets to JSONL files for training.")

    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input JSON files (default: every datasets/*.json)."
    )

    parser.add_argument(
        "--format",
        choices=["pairs", "conversations"],
        default="pairs",
        help="pairs: one .jsonl per dataset with {prompt, completion} lines (default). "
             "conversations: one .jsonl per conversation, one message per line, with "
             "<|response|>/<|end|>/<|branch|> markers (see conversation_format.py)."
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: .datasets for pairs, .conversations for conversations)."
    )

    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else ROOT / (
        ".conversations" if args.format == "conversations" else ".datasets"
    )
    inputs = args.inputs or [str(p) for p in sorted((ROOT / "datasets").glob("*.json"))]

    if not inputs:
        print("[ERROR] No input files found.")
        return

    if args.format == "pairs" and output_dir.is_dir():
        remove_legacy_outputs(output_dir)

    for input_file in inputs:
        input_path = Path(input_file)

        if not input_path.exists():
            print(f"[ERROR] File not found: {input_path}")
            continue

        try:
            convert_dataset(input_path, output_dir, args.format)
        except (json.JSONDecodeError, ValueError, OSError) as error:
            print(f"[ERROR] {input_path}: {error}")


if __name__ == "__main__":
    main()
