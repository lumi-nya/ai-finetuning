"""
Mensagem `system` compartilhada: a ficha da personagem + o catálogo de animações
+ as tools. Ela é igual em todas as conversas, então fica fora dos datasets
(datasets/*.json) e é juntada só quando o treino é gerado (convert_dataset.py)
ou quando o modelo é usado:

    character/lisa.txt     resumo da ficha (campo `character`)
    character/system.json  {"animation_catalog": [...], "tools": [...]}

Uma conversa que traga a sua PRÓPRIA mensagem `system` continua valendo (serve
de exceção); sem ela, entra a compartilhada.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHARACTER_PATH = ROOT / "character" / "lisa.txt"
SYSTEM_PATH = ROOT / "character" / "system.json"


def build_system(character, animation_catalog, tools):
    # ordem do treino: role, character, animation_catalog, tools
    return {"role": "system", "character": character, "animation_catalog": animation_catalog, "tools": tools}


def load_shared_system(character_path=None, system_path=None):
    character_path, system_path = character_path or CHARACTER_PATH, system_path or SYSTEM_PATH
    character = Path(character_path).read_text(encoding="utf-8").strip()
    data = json.loads(Path(system_path).read_text(encoding="utf-8"))
    for key in ("animation_catalog", "tools"):
        if not isinstance(data.get(key), list):
            raise ValueError(f"{system_path}: '{key}' precisa ser uma lista.")
    return build_system(character, data["animation_catalog"], data["tools"])


def save_shared_system(system, character_path=None, system_path=None):
    """Grava a ficha e o catálogo/tools nos dois arquivos (de forma atômica)."""
    character_path, system_path = character_path or CHARACTER_PATH, system_path or SYSTEM_PATH
    for path, text in (
        (character_path, str(system["character"]).strip() + "\n"),
        (system_path, json.dumps({"animation_catalog": system["animation_catalog"], "tools": system["tools"]},
                                 ensure_ascii=False, indent=4) + "\n"),
    ):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
        finally:
            if tmp.exists():
                tmp.unlink()


def has_system(conversation):
    return any(isinstance(m, dict) and m.get("role") == "system" for m in conversation)


def with_shared_system(conversation, shared):
    """Conversa sem `system` própria ganha a compartilhada no começo."""
    if has_system(conversation):
        return conversation
    return [json.loads(json.dumps(shared))] + list(conversation)


def without_shared_system(conversation, shared):
    """Tira a `system` quando é igual à compartilhada (para gravar o dataset sem repetição)."""
    if conversation and isinstance(conversation[0], dict) and conversation[0] == shared:
        return list(conversation[1:])
    return conversation
