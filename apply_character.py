#!/usr/bin/env python3
"""
Coloca o resumo da personagem (character/lisa.txt) no campo `character` da
mensagem `system` de todas as conversas de datasets/*.json. Pode rodar de novo
a qualquer momento (é idempotente): o campo é atualizado e fica logo depois de
"role", na ordem que o modelo vê no treino.

    python apply_character.py                       # character/lisa.txt, todos os datasets
    python apply_character.py --text outro.txt
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def with_character(system, text):
    out = {"role": system["role"], "character": text}
    out.update({k: v for k, v in system.items() if k not in ("role", "character")})
    return out


def apply(path, text):
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    changed = 0
    for conversation in data:
        for index, message in enumerate(conversation):
            if isinstance(message, dict) and message.get("role") == "system":
                new = with_character(message, text)
                if new != message or list(new) != list(message):
                    conversation[index] = new
                    changed += 1
    if changed:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=4) + ("\n" if raw.endswith("\n") else ""), encoding="utf-8")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", default=str(ROOT / "character" / "lisa.txt"))
    parser.add_argument("inputs", nargs="*", help="datasets (default: datasets/*.json)")
    args = parser.parse_args()
    text = Path(args.text).read_text(encoding="utf-8").strip()
    for path in [Path(p) for p in args.inputs] or sorted((ROOT / "datasets").glob("*.json")):
        print(f"{path.name}: {apply(path, text)} system atualizados")


if __name__ == "__main__":
    main()
