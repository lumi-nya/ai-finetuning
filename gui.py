#!/usr/bin/env python3
"""
GUI local para criar/editar os datasets e conversar com o modelo ajustado.

    python gui.py             # abre http://127.0.0.1:8765
    python gui.py --mock      # sem GPU: respostas simuladas, só para testar a interface

Usa apenas a biblioteca padrão. torch/transformers/peft só são importados
quando você clica em "Carregar modelo" na aba "Conversar" (via inference.py).

Os datasets são lidos e gravados em datasets/*.json, no mesmo formato que o
convert_dataset.py espera: uma lista de conversas, cada conversa uma lista de
mensagens (system, memory, user, mood, answer).
"""

import argparse
import gc
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import threading
import time
import traceback
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "gui.html"
DATASETS_DIR = ROOT / "datasets"

DEFAULT_MODEL = "Qwen/Qwen3-1.7B-Base"
DEFAULT_ADAPTER = "qwen-adapter"

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
MAX_BODY = 20 * 1024 * 1024
TOKEN = secrets.token_urlsafe(24)


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

def dataset_path(name):
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ApiError(400, "Nome inválido. Use letras, números, '-' e '_' (máx. 64).")
    return DATASETS_DIR / f"{name}.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_dataset(path):
    raw = path.read_bytes()
    return raw, json.loads(raw.decode("utf-8"))


def check_structure(data):
    """Mesmas exigências do convert_dataset.py."""
    if not isinstance(data, list):
        raise ApiError(400, "A raiz do JSON precisa ser uma lista de conversas.")
    for i, conv in enumerate(data, 1):
        if not isinstance(conv, list):
            raise ApiError(400, f"Conversa {i}: precisa ser uma lista de mensagens.")
        for j, msg in enumerate(conv, 1):
            if not isinstance(msg, dict):
                raise ApiError(400, f"Conversa {i}, mensagem {j}: precisa ser um objeto.")
            if "role" not in msg:
                raise ApiError(400, f"Conversa {i}, mensagem {j}: falta o campo 'role'.")


def normalize_for_save(data):
    """O JS não distingue 1 de 1.0; blend_weight é sempre float nos datasets."""
    for conv in data:
        for msg in conv:
            if msg.get("role") != "answer" or not isinstance(msg.get("timeline"), list):
                continue
            for ev in msg["timeline"]:
                if (
                    isinstance(ev, dict)
                    and ev.get("event_role") == "action"
                    and isinstance(ev.get("blend_weight"), int)
                    and not isinstance(ev.get("blend_weight"), bool)
                ):
                    ev["blend_weight"] = float(ev["blend_weight"])


def list_datasets():
    DATASETS_DIR.mkdir(exist_ok=True)
    items = []
    for path in sorted(DATASETS_DIR.glob("*.json")):
        if not NAME_RE.match(path.stem):
            continue
        entry = {"name": path.stem, "size": path.stat().st_size}
        try:
            _, data = read_dataset(path)
            if isinstance(data, list):
                entry["conversations"] = len(data)
            else:
                entry["error"] = "A raiz do JSON não é uma lista."
        except (OSError, ValueError) as error:
            entry["error"] = str(error)
        items.append(entry)
    return items


def load_dataset_api(name):
    path = dataset_path(name)
    if not path.is_file():
        raise ApiError(404, f"Arquivo não encontrado: datasets/{name}.json")
    try:
        raw, data = read_dataset(path)
    except (OSError, ValueError) as error:
        raise ApiError(422, f"Não consegui ler datasets/{name}.json: {error}")
    return {
        "name": name,
        "hash": sha(raw),
        "trailing_newline": raw.endswith(b"\n"),
        "data": data,
    }


def save_dataset_api(name, body):
    path = dataset_path(name)
    data = body.get("data")
    base_hash = body.get("base_hash")
    trailing = bool(body.get("trailing_newline", True))

    check_structure(data)

    if path.exists():
        if base_hash is None:
            raise ApiError(409, f"Já existe um arquivo chamado {name}.json.")
        if base_hash != sha(path.read_bytes()):
            raise ApiError(
                409,
                f"{name}.json mudou no disco depois que você o abriu.",
            )
    elif base_hash is not None:
        raise ApiError(409, f"{name}.json não existe mais no disco.")

    normalize_for_save(data)
    try:
        text = json.dumps(data, indent=4, ensure_ascii=False, allow_nan=False)
    except ValueError as error:
        raise ApiError(400, f"Valor inválido no JSON: {error}")
    raw = (text + ("\n" if trailing else "")).encode("utf-8")

    DATASETS_DIR.mkdir(exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_bytes(raw)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {"hash": sha(raw), "size": len(raw)}


def compute_defaults():
    """Catálogo, emoções e agent_id mais comuns nos datasets existentes."""
    catalogs, emotions, agents = Counter(), Counter(), Counter()
    catalog_by_key = {}
    for path in sorted(DATASETS_DIR.glob("*.json")):
        try:
            _, data = read_dataset(path)
        except (OSError, ValueError):
            continue
        if not isinstance(data, list):
            continue
        for conv in data:
            if not isinstance(conv, list):
                continue
            for msg in conv:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                if role == "system" and isinstance(msg.get("animation_catalog"), list):
                    key = json.dumps(msg["animation_catalog"], ensure_ascii=False)
                    catalogs[key] += 1
                    catalog_by_key[key] = msg["animation_catalog"]
                elif role == "mood" and isinstance(msg.get("primary_emotion"), str):
                    emotions[msg["primary_emotion"]] += 1
                elif role == "answer" and isinstance(msg.get("agent_id"), str):
                    agents[msg["agent_id"]] += 1
    catalog = catalog_by_key[catalogs.most_common(1)[0][0]] if catalogs else []
    return {
        "catalog": catalog,
        "emotions": sorted(e for e in emotions if e),
        "agent_id": agents.most_common(1)[0][0] if agents else "Lisa",
    }


# --------------------------------------------------------------------------
# Saída do modelo
# --------------------------------------------------------------------------

_decoder = json.JSONDecoder()


def parse_model_output(text):
    """
    O modelo continua o prompt (um objeto JSON por linha). Extrai os objetos que
    ele gerou e descarta o que vier depois da resposta (ex.: um 'user' inventado).
    """
    objects, junk = [], []
    pos, n = 0, len(text)
    while pos < n:
        start = text.find("{", pos)
        if start == -1:
            if text[pos:].strip():
                junk.append(text[pos:].strip())
            break
        if text[pos:start].strip():
            junk.append(text[pos:start].strip())
        try:
            obj, end = _decoder.raw_decode(text, start)
        except ValueError:
            newline = text.find("\n", start)
            stop = n if newline == -1 else newline
            junk.append(text[start:stop].strip())
            pos = stop
            continue
        objects.append(obj)
        pos = end

    mood = answer = None
    extra = []
    stopped_at = None
    for obj in objects:
        role = obj.get("role") if isinstance(obj, dict) else None
        if role in ("user", "system"):
            stopped_at = role
            break
        if role == "mood" and mood is None and answer is None:
            mood = obj
        elif role == "answer" and answer is None:
            answer = obj
        else:
            extra.append(obj)
    return {
        "mood": mood,
        "answer": answer,
        "extra": extra,
        "junk": junk,
        "stopped_at": stopped_at,
    }


def build_prompt(context):
    # Mesma serialização do LisaInference.generate (só para exibir na GUI):
    # sem mensagens 'mood', uma mensagem JSON por linha, terminando em "\n".
    items = [item for item in context if item.get("role") != "mood"]
    return "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n"


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------

class MockModel:
    """Substituto sem GPU: devolve algo no formato que o modelo ajustado gera."""

    def generate(self, context, max_new_tokens=256):
        time.sleep(0.6)
        animations = []
        for item in context:
            if isinstance(item, dict) and item.get("role") == "system":
                animations = [
                    a.get("id") for a in item.get("animation_catalog", []) if isinstance(a, dict)
                ]
        last_user = next(
            (m.get("text", "") for m in reversed(context) if m.get("role") == "user"), ""
        )
        timeline = []
        if animations:
            timeline.append(
                {"event_role": "action", "animation_id": animations[0], "blend_weight": 0.85}
            )
        timeline.append({"event_role": "speech", "text": f"(simulado) Você disse: {last_user[:60]}"})
        timeline.append({"event_role": "speech", "text": "Isto é só o modo de teste."})
        mood = {
            "role": "mood",
            "primary_emotion": "curious",
            "shyness_level": 3,
            "comfort_level": 7,
            "distraction_level": 2,
            "energy_level": 6,
        }
        mood_delta = {
            "primary_emotion": "amused",
            "shyness_level": -1,
            "comfort_level": 1,
            "distraction_level": 0,
            "energy_level": 2,
        }
        answer = {"role": "answer", "agent_id": "Lisa", "timeline": timeline, "mood_delta": mood_delta}
        invented = {"role": "user", "text": "texto inventado que deve ser descartado"}
        return "\n".join(json.dumps(o, ensure_ascii=False) for o in (mood, answer, invented))


class ModelManager:
    def __init__(self, mock, model_name, adapter):
        self.mock = mock
        self.model_name = model_name
        self.adapter = adapter
        self.state = "idle"  # idle | loading | ready | error
        self.error = None
        self.model = None
        self.lock = threading.Lock()
        self.gen_lock = threading.Lock()

    def status(self):
        with self.lock:
            return {
                "state": self.state,
                "error": self.error,
                "mock": self.mock,
                "model": self.model_name,
                "adapter": self.adapter,
            }

    def load(self, model_name, adapter):
        with self.lock:
            if self.state == "loading":
                raise ApiError(409, "O modelo já está sendo carregado.")
            self.state = "loading"
            self.error = None
            self.model_name = model_name
            self.adapter = adapter
        threading.Thread(target=self._load, args=(model_name, adapter), daemon=True).start()

    def _load(self, model_name, adapter):
        try:
            with self.gen_lock:
                self.model = None
                gc.collect()
                if self.mock:
                    time.sleep(1.0)
                    model = MockModel()
                else:
                    adapter_path = Path(adapter)
                    if not adapter_path.is_absolute():
                        adapter_path = ROOT / adapter_path
                    if not adapter_path.is_dir():
                        raise FileNotFoundError(
                            f"Adapter não encontrado em {adapter_path}. "
                            "Rode train_model.py primeiro ou informe o caminho certo."
                        )
                    if str(ROOT) not in sys.path:
                        sys.path.insert(0, str(ROOT))
                    from inference import LisaInference  # importa torch/transformers

                    model = LisaInference(model_name=model_name, adapter_path=str(adapter_path))
            with self.lock:
                self.model = model
                self.state = "ready"
        except BaseException as error:  # noqa: BLE001 - qualquer falha vira status de erro
            traceback.print_exc()
            with self.lock:
                self.state = "error"
                self.error = f"{type(error).__name__}: {error}"

    def generate(self, context, max_new_tokens):
        with self.lock:
            model, state = self.model, self.state
        if state != "ready" or model is None:
            raise ApiError(409, "O modelo ainda não está carregado.")
        with self.gen_lock:
            started = time.time()
            raw = model.generate(context, max_new_tokens=max_new_tokens)
            return raw, time.time() - started


MODEL = None  # criado em main()


def chat_api(body):
    context = body.get("context")
    if not isinstance(context, list) or not all(isinstance(m, dict) for m in context):
        raise ApiError(400, "context precisa ser uma lista de objetos.")
    try:
        max_new_tokens = int(body.get("max_new_tokens", 256))
    except (TypeError, ValueError):
        raise ApiError(400, "max_new_tokens precisa ser um número.")
    max_new_tokens = max(16, min(1024, max_new_tokens))
    raw, seconds = MODEL.generate(context, max_new_tokens)
    result = parse_model_output(raw)
    result.update({"raw": raw, "seconds": round(seconds, 2), "prompt": build_prompt(context)})
    return result


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "LisaGUI/1.0"

    def log_message(self, fmt, *args):
        if "--verbose" in sys.argv:
            super().log_message(fmt, *args)

    # -- infra
    def _send(self, status, payload, content_type="application/json; charset=utf-8", extra=None):
        body = payload if isinstance(payload, bytes) else json.dumps(
            payload, ensure_ascii=False
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("application/json"):
            raise ApiError(415, "Envie application/json.")
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ApiError(413, "Corpo grande demais.")
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8") or "null")
        except ValueError:
            raise ApiError(400, "JSON inválido.")
        if not isinstance(body, dict):
            raise ApiError(400, "O corpo precisa ser um objeto JSON.")
        return body

    def _guard(self):
        if self.headers.get("Host", "") not in self.server.allowed_hosts:
            raise ApiError(403, "Host não permitido.")

    def _dispatch(self, method):
        try:
            self._guard()
            path = self.path.split("?", 1)[0]
            if method == "GET" and path in ("/", "/index.html"):
                return self._serve_html()
            if not path.startswith("/api/"):
                raise ApiError(404, "Não encontrado.")
            supplied = self.headers.get("X-GUI-Token", "")
            if not hmac.compare_digest(supplied, TOKEN):
                raise ApiError(403, "Token inválido. Recarregue a página.")
            result = self._route(method, path)
            self._send(200, result)
        except ApiError as error:
            self._send(error.status, {"error": error.message})
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"error": f"{type(error).__name__}: {error}"})

    def _serve_html(self):
        if not HTML_PATH.is_file():
            raise ApiError(500, "gui.html não está ao lado de gui.py.")
        html = HTML_PATH.read_text(encoding="utf-8").replace("__GUI_TOKEN__", TOKEN)
        csp = (
            "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
            "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'self'; "
            "frame-ancestors 'none'"
        )
        self._send(
            200,
            html.encode("utf-8"),
            "text/html; charset=utf-8",
            {"Content-Security-Policy": csp},
        )

    def _route(self, method, path):
        if method == "GET" and path == "/api/datasets":
            return {"datasets": list_datasets()}
        if method == "GET" and path == "/api/defaults":
            return compute_defaults()
        if path.startswith("/api/datasets/"):
            name = path[len("/api/datasets/"):]
            if method == "GET":
                return load_dataset_api(name)
            if method == "PUT":
                return save_dataset_api(name, self._read_json())
        if method == "GET" and path == "/api/model":
            return MODEL.status()
        if method == "POST" and path == "/api/model/load":
            body = self._read_json()
            model_name = str(body.get("model") or MODEL.model_name).strip()
            adapter = str(body.get("adapter") or MODEL.adapter).strip()
            MODEL.load(model_name, adapter)
            return MODEL.status()
        if method == "POST" and path == "/api/chat":
            return chat_api(self._read_json())
        raise ApiError(404, "Rota não encontrada.")

    def do_GET(self):
        self._dispatch("GET")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_POST(self):
        self._dispatch("POST")


def main():
    global MODEL

    parser = argparse.ArgumentParser(description="GUI para os datasets e o chat com a Lisa.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mock", action="store_true", help="respostas simuladas, sem GPU")
    parser.add_argument("--no-browser", action="store_true", help="não abrir o navegador")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="modelo base (padrão: %(default)s)")
    parser.add_argument(
        "--adapter", default=DEFAULT_ADAPTER, help="pasta do adapter LoRA (padrão: %(default)s)"
    )
    parser.add_argument("--verbose", action="store_true", help="loga cada requisição")
    args = parser.parse_args()

    MODEL = ModelManager(args.mock, args.model, args.adapter)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.allowed_hosts = {f"127.0.0.1:{args.port}", f"localhost:{args.port}"}

    url = f"http://127.0.0.1:{args.port}/"
    print(f"GUI em {url}  (Ctrl+C para sair)")
    if args.mock:
        print("Modo de teste: o chat devolve respostas simuladas.")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
