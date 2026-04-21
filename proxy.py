#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import aiohttp
import yaml
from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("llama-proxy")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

_config = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f) or {}

PROXY_HOST = _config.get("proxy", {}).get("host", "0.0.0.0")
PROXY_PORT = _config.get("proxy", {}).get("port", 8888)
NUM_SLOTS = _config.get("proxy", {}).get("slots", 4)
LLAMA_URL = _config.get("llama_server", {}).get("url", "http://10.0.0.20:11400")


@dataclass
class SlotInfo:
    slot_id: int
    session_id: Optional[str] = None
    msg_count: int = 0
    char_count: int = 0
    last_used: float = field(default_factory=time.time)
    request_count: int = 0
    processing: bool = False


@dataclass
class SessionInfo:
    session_id: str
    slot_id: int
    msg_count: int = 0
    char_count: int = 0
    last_used: float = field(default_factory=time.time)


def get_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content)


def make_session_id(messages: list) -> str:
    extracted = [
        {"role": m.get("role", ""), "content": get_text_content(m.get("content", ""))}
        for m in messages[:3]
    ]
    return hashlib.sha256(json.dumps(extracted, sort_keys=True).encode()).hexdigest()[
        :16
    ]


def count_chars(messages: list) -> int:
    return sum(len(get_text_content(m.get("content", ""))) for m in messages)


class LlamaProxy:
    def __init__(self, num_slots: int):
        self.num_slots = num_slots
        self.slots: Dict[int, SlotInfo] = {
            i: SlotInfo(slot_id=i) for i in range(num_slots)
        }
        self.sessions: OrderedDict[str, SessionInfo] = OrderedDict()
        self.lock = asyncio.Lock()
        self._http: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(num_slots)
        self.waiting_requests = 0

    @property
    def http(self) -> aiohttp.ClientSession:
        if not self._http or self._http.closed:
            self._http = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=20),
                timeout=aiohttp.ClientTimeout(total=None, connect=10),
            )
        return self._http

    def _get_eviction_score(self, session: SessionInfo, now: float) -> float:
        return ((now - session.last_used) * 1000) / (session.char_count + 1000)

    def _get_slot_for_session(self, session_id: str) -> int:
        if session_id in self.sessions:
            self.sessions.move_to_end(session_id)
            return self.sessions[session_id].slot_id

        for slot in self.slots.values():
            if slot.session_id is None:
                return slot.slot_id

        if not self.sessions:
            return 0

        now = time.time()
        best_evict_id = max(
            self.sessions.keys(),
            key=lambda sid: self._get_eviction_score(self.sessions[sid], now),
        )
        evict_slot_id = self.sessions[best_evict_id].slot_id

        del self.sessions[best_evict_id]
        self.slots[evict_slot_id].session_id = None
        return evict_slot_id

    def _bind_session(self, session_id: str, slot_id: int, messages: list):
        now = time.time()
        slot = self.slots[slot_id]
        slot.session_id = session_id
        slot.msg_count = len(messages)
        slot.char_count = count_chars(messages)
        slot.last_used = now
        slot.request_count += 1

        if session_id in self.sessions:
            s = self.sessions[session_id]
            s.msg_count = slot.msg_count
            s.char_count = slot.char_count
            s.last_used = now
            self.sessions.move_to_end(session_id)
        else:
            self.sessions[session_id] = SessionInfo(
                session_id=session_id,
                slot_id=slot_id,
                msg_count=slot.msg_count,
                char_count=slot.char_count,
                last_used=now,
            )

    async def handle_chat(self, request: web.Request) -> web.StreamResponse:
        self.waiting_requests += 1
        try:
            await self.semaphore.acquire()
        except asyncio.CancelledError:
            self.waiting_requests -= 1
            raise

        self.waiting_requests -= 1
        try:
            try:
                body = await request.json()
            except Exception:
                raise web.HTTPBadRequest(reason="Invalid JSON")

            messages = body.get("messages", [])
            body["cache_prompt"] = True
            session_id = make_session_id(messages)

            async with self.lock:
                slot_id = self._get_slot_for_session(session_id)
                self._bind_session(session_id, slot_id, messages)
                body["id_slot"] = slot_id
                self.slots[slot_id].processing = True
                log.info(
                    f"session={session_id[:8]} -> slot={slot_id} msgs={len(messages)}"
                )

            try:
                return await self._forward_request(request, body)
            finally:
                try:
                    await asyncio.shield(asyncio.sleep(0.3))
                except Exception:
                    pass
                self.slots[slot_id].processing = False
        finally:
            self.semaphore.release()

    async def _forward_request(
        self, original_request: web.Request, body: dict
    ) -> web.StreamResponse:
        url = f"{LLAMA_URL}/v1/chat/completions"
        headers = {
            k: v
            for k, v in original_request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }
        headers["Content-Type"] = "application/json"

        if body.get("stream", True):
            resp = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
            await resp.prepare(original_request)
            try:
                async with self.http.post(url, json=body, headers=headers) as upstream:
                    async for chunk in upstream.content.iter_any():
                        await resp.write(chunk)
                await resp.write_eof()
            except Exception as e:
                log.info(f"Stream interrupted: {e}")
            return resp
        else:
            try:
                async with self.http.post(url, json=body, headers=headers) as upstream:
                    data = await upstream.read()
                    return web.Response(
                        status=upstream.status,
                        headers={"Content-Type": "application/json"},
                        body=data,
                    )
            except Exception as e:
                log.error(f"Forwarding error: {e}")
                return web.Response(status=500, text=str(e))

    async def handle_passthrough(self, request: web.Request) -> web.Response:
        url = f"{LLAMA_URL}{request.path_qs}"
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length")
        }
        body = await request.read()
        async with self.http.request(
            request.method, url, headers=headers, data=body
        ) as resp:
            data = await resp.read()
            return web.Response(
                status=resp.status, headers=dict(resp.headers), body=data
            )

    async def handle_status(self, request: web.Request) -> web.Response:
        now = time.time()
        slots_data = []
        for s in self.slots.values():
            score = 0.0
            if s.session_id and s.session_id in self.sessions:
                score = self._get_eviction_score(self.sessions[s.session_id], now)
            slots_data.append(
                {
                    "slot_id": s.slot_id,
                    "session_id": s.session_id,
                    "msg_count": s.msg_count,
                    "char_count": s.char_count,
                    "requests": s.request_count,
                    "processing": s.processing,
                    "evict_score": round(score, 2),
                }
            )

        return web.json_response(
            {
                "num_slots": self.num_slots,
                "waiting_requests": self.waiting_requests,
                "slots": slots_data,
            }
        )


def main():
    global LLAMA_URL
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-host", type=str, default=PROXY_HOST)
    parser.add_argument("--proxy-port", type=int, default=PROXY_PORT)
    parser.add_argument("--llama-url", type=str, default=LLAMA_URL)
    parser.add_argument("--slots", type=int, default=NUM_SLOTS)
    args = parser.parse_args()

    LLAMA_URL = args.llama_url

    proxy = LlamaProxy(num_slots=args.slots)
    app = web.Application(client_max_size=200 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", proxy.handle_chat)
    app.router.add_get("/proxy/status", proxy.handle_status)
    app.router.add_route("*", "/{path_info:.*}", proxy.handle_passthrough)

    log.info(
        f"Proxy started: {args.proxy_host}:{args.proxy_port} -> {args.llama_url} slots={args.slots}"
    )
    web.run_app(
        app, host=args.proxy_host, port=args.proxy_port, print=None, access_log=None
    )


if __name__ == "__main__":
    main()
