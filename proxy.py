import json
import re

import requests
from flask import Flask, Response, request

app = Flask(__name__)
LLAMA_URL = "http://10.0.0.20:11400"


# 从 inbound meta 块里提取 channel 信息以分配 slot
def detect_slot_from_meta(body):
    messages = body.get("messages", [])
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")

            # 从 inbound meta 块里提取 channel 信息
            pattern = r'(?s)##\s*Inbound Context.*?\n([^\n]*?"channel"[^\n]*)'
            match = re.search(pattern, content, re.IGNORECASE).group(1).strip()

            if "discord" in match.lower():
                return 1
            if "webchat" in match.lower():
                return 2
            return 0
    return 0


@app.route("/<path:path>", methods=["POST"])
@app.route("/v1/<path:path>", methods=["POST"])
def proxy(path):
    body = request.get_json()
    if not body:
        return Response(status=400)

    # 1. 检测来源，注入 id_slot
    slot_id = detect_slot_from_meta(body)
    body["id_slot"] = slot_id
    print(f"[proxy] path={path} slot={slot_id}", flush=True)

    # 2. 透传 streaming
    is_stream = body.get("stream", False)
    upstream = requests.post(
        f"{LLAMA_URL}/v1/{path}", json=body, stream=is_stream, timeout=300
    )

    if is_stream:

        def generate():
            for chunk in upstream.iter_content(chunk_size=None):
                yield chunk

        return Response(
            generate(),
            status=upstream.status_code,
            content_type=upstream.headers.get("Content-Type", "text/event-stream"),
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Transfer-Encoding": "chunked",
            },
        )
    else:
        return Response(
            upstream.content,
            status=upstream.status_code,
            content_type=upstream.headers.get("Content-Type", "application/json"),
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888, threaded=True)
