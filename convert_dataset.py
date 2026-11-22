
import json
import argparse
from pathlib import Path

from prompting import GENERATED_ROLES, render_prompt

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
    prompt/completion. Um turno simples tem um passo (mood + answer); um turno
    com tool calling tem dois ou mais:

      user -> [tool_call+] -> tool_result+ -> [tool_call+ -> tool_result+] -> mood + answer

    Cada sequência seguida de mensagens do modelo (tool_call, mood, answer)
    vira uma completion; as mensagens externas (user, tool_result) só entram
    no prompt do passo seguinte. O prompt é montado por prompting.render_prompt,
    igual na inferência: sem mood/branch, e sem tool_call/tool_result dos
    turnos passados.

    Um turno pode ter respostas alternativas (mensagens 'branch' com 'mood' e
    'answer'): cada uma vira um exemplo extra com o prompt do passo final, e o
    modelo aprende que aquele turno admite vários humores. Os ramos não entram
    no contexto dos turnos seguintes.

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
        positions = [
            i for i in range(start + 1, end) if conversation[i]["role"] != "branch"
        ]
        branches = [
            conversation[i] for i in range(start + 1, end) if conversation[i]["role"] == "branch"
        ]

        if not positions and not branches:
            print(f"[WARN] {label}, turn {turn} ignored: nothing after the user message.")
            continue

        def prefix(upto):
            return [conversation[i] for i in range(upto) if conversation[i]["role"] != "branch"]

        final_prompt = None
        pending_calls = 0
        k = 0
        while k < len(positions):
            role = conversation[positions[k]]["role"]
            if role in GENERATED_ROLES:
                run = [positions[k]]
                while k + 1 < len(positions) and conversation[positions[k + 1]]["role"] in GENERATED_ROLES:
                    k += 1
                    run.append(positions[k])
                messages = [conversation[i] for i in run]
                prompt = render_prompt(prefix(run[0]))
                if final_prompt is None and any(m["role"] in ("mood", "answer") for m in messages):
                    final_prompt = prompt
                for m in messages:
                    if m["role"] == "tool_call":
                        pending_calls += 1
                        if tool_names and m.get("name") not in tool_names:
                            print(f"[WARN] {label}, turn {turn}: tool_call '{m.get('name')}' is not listed in system.tools.")
                        elif not tool_names:
                            print(f"[WARN] {label}, turn {turn}: tool_call but the system message has no 'tools'.")
                    elif m["role"] == "answer" and not isinstance(m.get("mood_delta"), dict):
                        print(f"[WARN] {label}, turn {turn}: answer without 'mood_delta' "
                              f"(the model would learn inconsistent labels).")
                if pending_calls and any(m["role"] in ("mood", "answer") for m in messages):
                    print(f"[WARN] {label}, turn {turn}: mood/answer generated before the tool_result.")
                examples.append({"prompt": prompt, "completion": "\n".join(to_lines(messages))})
            else:
                if role == "tool_result":
                    if pending_calls == 0:
                        print(f"[WARN] {label}, turn {turn}: tool_result without a tool_call before it.")
                    pending_calls = max(0, pending_calls - 1)
            k += 1

        if pending_calls:
            print(f"[WARN] {label}, turn {turn}: tool_call without a tool_result.")

        if final_prompt is None:
            final_prompt = render_prompt(prefix(end if not positions else positions[-1] + 1))

        for number, branch in enumerate(branches, start=2):
            mood, answer = branch.get("mood"), branch.get("answer")
            if not isinstance(mood, dict) or not isinstance(answer, dict):
                print(f"[WARN] {label}, turn {turn}, branch {number} ignored: needs 'mood' and 'answer' objects.")
                continue
            if not isinstance(answer.get("mood_delta"), dict):
                print(f"[WARN] {label}, turn {turn}, branch {number}: answer without 'mood_delta'.")
            examples.append({"prompt": final_prompt, "completion": "\n".join(to_lines([mood, answer]))})

    return examples


def convert_dataset(input_path: Path, output_dir: Path):
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
        "--output",
        default=str(ROOT / ".datasets"),
        help="Output directory (default: .datasets next to this script)."
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    inputs = args.inputs or [str(p) for p in sorted((ROOT / "datasets").glob("*.json"))]

    if not inputs:
        print("[ERROR] No input files found.")
        return

    if output_dir.is_dir():
        remove_legacy_outputs(output_dir)

    for input_file in inputs:
        input_path = Path(input_file)

        if not input_path.exists():
            print(f"[ERROR] File not found: {input_path}")
            continue

        try:
            convert_dataset(input_path, output_dir)
        except (json.JSONDecodeError, ValueError, OSError) as error:
            print(f"[ERROR] {input_path}: {error}")


if __name__ == "__main__":
    main()
