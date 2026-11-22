
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Nome usado por versões antigas deste script (um arquivo por conversa, uma
# mensagem por linha). Esses arquivos têm outro formato e precisam sair do
# caminho, senão o train_model.py mistura os dois.
LEGACY_PATTERN = "dataset-*.jsonl"


def to_lines(messages: list[dict]) -> list[str]:
    return [json.dumps(message, ensure_ascii=False) for message in messages]


def conversation_to_examples(conversation: list[dict], label: str) -> list[dict]:
    """
    Um exemplo de treino por turno, no formato prompt/completion:

      prompt     = tudo até a mensagem 'user' do turno, uma mensagem JSON por
                   linha, terminando em "\\n" (igual ao prompt de inference.py).
                   As mensagens 'mood' ficam de fora: dos turnos anteriores
                   só interessa o que foi dito e feito (user e answer).
      completion = o que o modelo deve gerar depois (mood + answer), até o
                   próximo 'user'

    O treino calcula a perda só na completion, então o mood de cada turno é
    aprendido como parte da resposta daquele turno. O prompt termina em "\\n"
    para que a fronteira prompt/completion caia entre dois tokens, do mesmo
    jeito no treino e na inferência.
    """
    user_positions = [
        i for i, message in enumerate(conversation) if message["role"] == "user"
    ]

    if not user_positions:
        print(f"[WARN] {label} ignored: no 'user' message.")
        return []

    examples = []

    for turn, start in enumerate(user_positions, start=1):
        is_last = turn == len(user_positions)
        end = len(conversation) if is_last else user_positions[turn]
        reply = conversation[start + 1:end]

        if not reply:
            print(f"[WARN] {label}, turn {turn} ignored: nothing after the user message.")
            continue

        context = [
            message for message in conversation[:start + 1]
            if message["role"] != "mood"
        ]

        examples.append({
            "prompt": "\n".join(to_lines(context)) + "\n",
            "completion": "\n".join(to_lines(reply)),
        })

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
