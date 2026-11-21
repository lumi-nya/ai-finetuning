
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Nome usado pela versão antiga deste script (um arquivo por conversa, uma
# mensagem por linha). Esses arquivos têm outro formato e precisam sair do
# caminho, senão o train_model.py mistura os dois.
LEGACY_PATTERN = "dataset-*.jsonl"


def conversation_to_text(conversation: list[dict]) -> str:
    """Uma mensagem JSON por linha, igual ao prompt montado em inference.py."""
    return "\n".join(
        json.dumps(message, ensure_ascii=False)
        for message in conversation
    )


def convert_dataset(input_path: Path, output_dir: Path):
    print(f"Processing: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError(f"{input_path}: the JSON root must be a list.")

    lines = []

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

        # Um exemplo de treino por conversa inteira, com o campo "text"
        # que o train_model.py lê (dataset_text_field="text").
        lines.append(
            json.dumps(
                {"text": conversation_to_text(conversation)},
                ensure_ascii=False
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Um arquivo por dataset de origem: datasets/playful.json -> .datasets/playful.jsonl
    output_path = output_dir / f"{input_path.stem}.jsonl"

    with output_path.open("w", encoding="utf-8") as output:
        for line in lines:
            output.write(line + "\n")

    print(f"  -> {output_path} ({len(lines)} conversations)")
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
