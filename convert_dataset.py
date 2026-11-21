
import json
import argparse
from pathlib import Path


def convert_dataset(input_path: Path, output_dir: Path):
    print(f"Processing: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError(f"{input_path}: the JSON root must be a list.")

    output_dir.mkdir(parents=True, exist_ok=True)

    for index, conversation in enumerate(dataset, start=1):
        if not isinstance(conversation, list):
            print(f"[WARN] Context {index} ignored: not a list.")
            continue

        output_path = output_dir / f"dataset-{index}.jsonl"

        with output_path.open("w", encoding="utf-8") as output:
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

                output.write(
                    json.dumps(message, ensure_ascii=False) + "\n"
                )

        print(f"  -> {output_path}")

    print(f"Finished: {input_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Convert JSON datasets to separated JSONL files.")

    parser.add_argument(
        "inputs",
        nargs="+",
        default="datasets.jsonl",
        help="Input JSON files."
    )

    parser.add_argument(
        "--output",
        default=".datasets",
        help="Exit directory (default: .datasets)."
    )

    args = parser.parse_args()

    output_dir = Path(args.output)

    for input_file in args.inputs:
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
