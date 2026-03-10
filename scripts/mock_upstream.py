#!/usr/bin/env python3
"""
Минимальный HTTP‑сервер, принимающий POST как UPSTREAM_URL (multipart или JSON).
Для полноценного теста без отправки на боевой бэкенд: API шлёт сюда, здесь логируем и отвечаем 200.

Запуск:
  python scripts/mock_upstream.py --port 9999
  или: uvicorn scripts.mock_upstream:app --host 127.0.0.1 --port 9999
Затем в app.env: UPSTREAM_URL=http://127.0.0.1:9999/events
"""
import argparse
import json
import sys
from typing import Any, Dict, List

# FastAPI уже есть в проекте
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()
EVENTS_RECEIVED: List[Dict[str, Any]] = []


@app.post("/events")
@app.post("/api/v1/anpr/events")
async def receive_event(request: Request):
    content_type = request.headers.get("content-type", "")
    entry: Dict[str, Any] = {"content_type": content_type}

    if "multipart" in content_type:
        form = await request.form()
        entry["form_keys"] = list(form.keys())
        event_str = form.get("event")
        entry["event_json"] = event_str
        if isinstance(event_str, str):
            try:
                entry["event"] = json.loads(event_str)
            except json.JSONDecodeError:
                entry["event"] = None
        else:
            entry["event"] = None
        files = []
        for key in form.keys():
            val = form[key]
            if hasattr(val, "filename") and val.filename:
                data = await val.read()
                files.append((key, val.filename, len(data)))
        entry["files"] = files
        EVENTS_RECEIVED.append(entry)
        print(f"[MOCK_UPSTREAM] multipart: form_keys={entry['form_keys']}, files={entry['files']}")
    else:
        try:
            body = await request.body()
            data = json.loads(body) if body else {}
            entry["json"] = data
            EVENTS_RECEIVED.append(entry)
            print(f"[MOCK_UPSTREAM] JSON: keys={list(data.keys()) if isinstance(data, dict) else 'n/a'}")
        except Exception:
            entry["raw"] = (await request.body()).decode("utf-8", errors="replace")[:500]
            EVENTS_RECEIVED.append(entry)
            print("[MOCK_UPSTREAM] body (not JSON)")

    return JSONResponse({"status": "ok", "received": len(EVENTS_RECEIVED)})


@app.get("/")
async def index():
    return {"mock": "upstream", "events_received": len(EVENTS_RECEIVED)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"[MOCK_UPSTREAM] Запуск: uvicorn scripts.mock_upstream:app --host {args.host} --port {args.port}")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
