#!/usr/bin/env python3
"""
llama-proxy: session → slot routing proxy

This proxy works with llama-server (llama.cpp server) and expects the server to be started
with the following recommended flags for correct behavior and reliable Prefill monitoring:

  --parallel 3          # number of slots, adjust as needed
  --kv-unified          # unified KV pool so system prompt is shared across slots
  --cache-ram 0         # disable memory-level cache (prevents clear_idle from silently clearing GPU KV)
  --ctx-size 65536      # total KV pool size

Design:
  - Each session_id is bound to a fixed slot.
  - Requests for the same session always go to the same slot so KV cache accumulates.
  - Session ID is computed from the full text of the first 3 messages to ensure stable prefix matching.
  - Cost-aware eviction: take idle time and recompute cost (character count) into account so long/expensive
    conversations are less likely to be evicted.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
import yaml
from aiohttp import web
from source_detector import detect_source

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("llama-proxy")

# ─────────────────────────────────────────────
# Global configuration
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)

PROXY_HOST = _config.get("proxy", {}).get("host", "0.0.0.0")
PROXY_PORT = _config.get("proxy", {}).get("port", 8888)
NUM_SLOTS = _config.get("proxy", {}).get("slots", 4)
LLAMA_URL = _config.get("llama_server", {}).get("url", "http://10.0.0.20:11400")


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────


@dataclass
class SlotInfo:
    slot_id: int
    session_id: Optional[str] = None
    msg_count: int = 0
    char_count: int = 0
    last_used: float = field(default_factory=time.time)
    request_count: int = 0
    processing: bool = False
    latest_token: str = ""
    matched_chars: int = 0
    new_chars: int = 0
    source: str = ""

    def is_idle(self) -> bool:
        return self.session_id is None


@dataclass
class SessionInfo:
    session_id: str
    slot_id: int
    msg_count: int = 0
    char_count: int = 0
    first_seen: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    request_count: int = 0


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────


def make_session_id(messages: list) -> str:
    """
    Generate a stable session ID based on message content.

    We extract the full plain-text content of the first 3 messages and hash that,
    avoiding collisions caused by naive truncation.
    """
    extracted = []
    for m in messages[:3]:
        role = m.get("role", "")
        content = m.get("content", "")

        text_content = ""
        if isinstance(content, str):
            text_content = content
        elif isinstance(content, list):
            # Handle multimodal structures: extract only text parts
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_content += part.get("text", "")
        else:
            text_content = str(content)

        extracted.append({"role": role, "content": text_content})

    key = json.dumps(extracted, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def common_prefix_len(a: list, b: list) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        ra = {"role": a[i].get("role", ""), "content": a[i].get("content", "")}
        rb = {"role": b[i].get("role", ""), "content": b[i].get("content", "")}
        if ra != rb:
            return i
    return n


def count_chars(messages: list) -> int:
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(part.get("text", ""))
        else:
            total += len(str(content))
    return total


# ─────────────────────────────────────────────
# Proxy core
# ─────────────────────────────────────────────


class LlamaProxy:
    def __init__(self, num_slots: int):
        self.num_slots = num_slots
        self.slots: dict[int, SlotInfo] = {
            i: SlotInfo(slot_id=i) for i in range(num_slots)
        }
        self.sessions: OrderedDict[str, SessionInfo] = OrderedDict()
        self.slot_messages: dict[int, list] = {i: [] for i in range(num_slots)}
        self.lock = asyncio.Lock()
        self._http: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(num_slots)
        self.waiting_requests: int = 0
        self._token_started: dict[int, bool] = {i: False for i in range(num_slots)}
        log.info(f"Initialized: {num_slots} slots")

    @property
    def http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            connector = aiohttp.TCPConnector(limit=20)
            timeout = aiohttp.ClientTimeout(total=None, connect=10)
            self._http = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self._http

    def _get_slot_for_session(self, session_id: str, messages: list) -> tuple[int, str]:
        # 1. Existing binding
        if session_id in self.sessions:
            slot_id = self.sessions[session_id].slot_id
            self.sessions.move_to_end(session_id)
            return slot_id, "reusing existing slot"

        # 2. Idle slot available
        for slot in self.slots.values():
            if slot.is_idle():
                return slot.slot_id, "assigning idle slot"

        # 3. Cost-Aware eviction
        if not self.sessions:
            # Prevent crashes in cases where sessions is empty but slots are still marked occupied
            return 0, "force reuse slot 0 (error recovery)"

        now = time.time()
        best_evict_session_id = None
        max_evict_score = -1.0

        for sid, sinfo in self.sessions.items():
            idle_time = now - sinfo.last_used
            # Eviction score = (idle_time_seconds * 1000) / (char_count + 1000)
            # Larger char_count -> larger denominator -> lower score -> less likely to be evicted
            score = (idle_time * 1000) / (sinfo.char_count + 1000)
            if score > max_evict_score:
                max_evict_score = score
                best_evict_session_id = sid

        evict_session_id = best_evict_session_id or next(iter(self.sessions))
        evict_slot_id = self.sessions[evict_session_id].slot_id

        del self.sessions[evict_session_id]
        self.slots[
            evict_slot_id
        ].session_id = None  # prevent leftover state if forward fails later
        return (
            evict_slot_id,
            f"evicting session {evict_session_id[:8]} (score: {max_evict_score:.1f})",
        )

    def _bind_session(self, session_id: str, slot_id: int, messages: list, source: str):
        now = time.time()
        msg_count = len(messages)

        char_count = count_chars(messages)

        slot = self.slots[slot_id]
        slot.session_id = session_id
        slot.msg_count = msg_count
        slot.char_count = char_count
        slot.last_used = now
        slot.request_count += 1
        slot.source = source

        if session_id in self.sessions:
            info = self.sessions[session_id]
            info.msg_count = msg_count
            info.char_count = char_count
            info.last_used = now
            info.request_count += 1
        else:
            self.sessions[session_id] = SessionInfo(
                session_id=session_id,
                slot_id=slot_id,
                msg_count=msg_count,
                char_count=char_count,
                last_used=now,
                request_count=1,
            )
            self.sessions.move_to_end(session_id)

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

            messages: list = body.get("messages", [])
            stream: bool = body.get("stream", True)

            # Detect request source (using a separate module)
            source = detect_source(messages, body, dict(request.headers))
            # Debug: capture requests that couldn't be confidently identified to help refine regexes
            if source in ["OpenAI", "Unknown", "Python", "Python-Req", "AioHttp"]:
                try:
                    debug_log_path = os.path.join(BASE_DIR, "debug_source.log")
                    with open(debug_log_path, "a", encoding="utf-8") as f_debug:
                        f_debug.write(
                            f"\n--- [{time.strftime('%H:%M:%S')}] Source={source} ---\n"
                        )
                        f_debug.write(f"UA: {request.headers.get('User-Agent')}\n")
                        f_debug.write(f"Body User: {body.get('user')}\n")
                        f_debug.write(
                            f"First Msg: {json.dumps(messages[0] if messages else {}, ensure_ascii=False)}\n"
                        )
                        sys_p = next(
                            (
                                m.get("content")
                                for m in messages
                                if m.get("role") == "system"
                            ),
                            "None",
                        )
                        f_debug.write(f"Full System Prompt: {sys_p}\n")
                except:
                    pass

            # Force enable cache handling on the body
            body["cache_prompt"] = True
            session_id = make_session_id(messages)

            async with self.lock:
                slot_id, reason = self._get_slot_for_session(session_id, messages)

                # Analyze cache reuse
                cached_msgs = self.slot_messages.get(slot_id, [])
                prefix_len = common_prefix_len(messages, cached_msgs)

                matched_chars = count_chars(messages[:prefix_len])
                total_chars = count_chars(messages)
                new_chars = total_chars - matched_chars

                log.info(
                    f"session={session_id[:8]} -> slot={slot_id} | {reason} | "
                    f"msgs={len(messages)} match={prefix_len} new_chars={new_chars}"
                )

                body["id_slot"] = slot_id

                # Pre-update occupancy state
                self._bind_session(session_id, slot_id, messages, source)
                self.slots[slot_id].processing = True
                self._token_started[slot_id] = False
                self.slots[slot_id].matched_chars = matched_chars
                self.slots[slot_id].new_chars = new_chars

            try:
                response, success = await self._forward(request, body, stream, slot_id)
                if success:
                    # Only commit the index for fully successful requests
                    self.slot_messages[slot_id] = list(messages)
                else:
                    log.warning(f"Slot {slot_id} request incomplete, resetting index")
                    self.slot_messages[slot_id] = []
                return response
            except (Exception, asyncio.CancelledError) as e:
                log.info(f"Client interrupted ({type(e).__name__}) - index cleared")
                self.slot_messages[slot_id] = []
                raise
            finally:
                # Core safeguard: use shield to ensure a short delay is not cancelled so the monitor can grab the last frame
                try:
                    await asyncio.shield(asyncio.sleep(0.3))
                except:
                    pass
                self.slots[slot_id].processing = False
        finally:
            self.semaphore.release()

    async def _forward(
        self, original_request: web.Request, body: dict, stream: bool, slot_id: int
    ) -> tuple[web.StreamResponse, bool]:
        url = f"{LLAMA_URL}/v1/chat/completions"
        headers = {
            k: v
            for k, v in original_request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }
        headers["Content-Type"] = "application/json"
        timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=600)

        if stream:
            resp = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
            try:
                await resp.prepare(original_request)
                async with self.http.post(
                    url, json=body, headers=headers, timeout=timeout
                ) as upstream:
                    buffer = b""
                    async for chunk in upstream.content.iter_any():
                        await resp.write(chunk)
                        buffer += chunk
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            line = line.strip()
                            if line.startswith(b"data:") and b"[DONE]" not in line:
                                try:
                                    payload = line.split(b":", 1)[1].strip()
                                    j = json.loads(payload.decode("utf-8"))
                                    delta = j.get("choices", [{}])[0].get("delta", {})
                                    content = (delta.get("content") or "") + (
                                        delta.get("reasoning_content") or ""
                                    )
                                    if content:
                                        if not self._token_started[slot_id]:
                                            self.slots[slot_id].latest_token = ""
                                            self._token_started[slot_id] = True
                                        self.slots[slot_id].latest_token = (
                                            self.slots[slot_id].latest_token
                                            + content.replace("\n", " ")
                                        )[-60:]
                                except Exception:
                                    pass

                    # Handle trailing buffer or a final line without newline
                    last_line = buffer.strip()
                    if last_line.startswith(b"data:") and b"[DONE]" not in last_line:
                        try:
                            payload = last_line.split(b":", 1)[1].strip()
                            j = json.loads(payload.decode("utf-8"))
                            delta = j.get("choices", [{}])[0].get("delta", {})
                            content = (delta.get("content") or "") + (
                                delta.get("reasoning_content") or ""
                            )
                            if content:
                                if not self._token_started[slot_id]:
                                    self.slots[slot_id].latest_token = ""
                                    self._token_started[slot_id] = True
                                self.slots[slot_id].latest_token = (
                                    self.slots[slot_id].latest_token
                                    + content.replace("\n", " ")
                                )[-60:]
                        except:
                            pass
                await resp.write_eof()
                return resp, True
            except (Exception, asyncio.CancelledError) as e:
                # Ignore write errors caused by client disconnects; return response to avoid aiohttp 500
                log.info(
                    f"Stream transfer interrupted (client disconnected): {type(e).__name__}"
                )
                return resp, False
        else:
            # For non-stream requests (e.g., TitleGen), use an aggressive JSON deep-search parsing strategy
            async def fetch_and_parse():
                async with self.http.post(
                    url, json=body, headers=headers, timeout=timeout
                ) as upstream:
                    raw_data = await upstream.read()
                    try:
                        resp_text = raw_data.decode("utf-8", errors="ignore")
                        j = json.loads(resp_text)

                        # Breadth-first search for likely text sources
                        res = ""
                        choices = j.get("choices", [])
                        if choices:
                            m = choices[0].get("message", {})
                            res = (
                                m.get("content")
                                or m.get("reasoning_content")
                                or choices[0].get("text")
                                or ""
                            )
                        else:
                            res = j.get("content") or j.get("text") or ""

                        if not res:
                            # Bruteforce search for any string > 2 chars that is not obvious metadata
                            for key, val in j.items():
                                if (
                                    isinstance(val, str)
                                    and len(val) > 2
                                    and key not in ["id", "model", "object"]
                                ):
                                    res = val
                                    break

                        if res:
                            self.slots[slot_id].latest_token = (
                                str(res)
                                .replace("\n", " ")
                                .replace("  ", " ")
                                .strip()[-60:]
                            )
                            self._token_started[slot_id] = True
                            log.info(
                                f"Slot {slot_id} non-stream parse success: {self.slots[slot_id].latest_token}"
                            )
                    except Exception as e:
                        log.error(f"Slot {slot_id} non-stream parse failed: {e}")
                    return raw_data, upstream.status

            try:
                # Even if the client disconnects early (e.g., Hermes closes after getting a title),
                # use shield() to ensure we read and parse the upstream response fully.
                data, status = await asyncio.shield(fetch_and_parse())
                return web.Response(
                    status=status,
                    headers={"Content-Type": "application/json"},
                    body=data,
                ), (status == 200)
            except Exception as e:
                log.error(f"Slot {slot_id} forwarding error: {e}")
                return web.Response(status=500), False

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
                sinfo = self.sessions[s.session_id]
                idle_time = now - sinfo.last_used
                score = (idle_time * 1000) / (sinfo.char_count + 1000)

            slots_data.append(
                {
                    "slot_id": s.slot_id,
                    "session_id": s.session_id,
                    "msg_count": s.msg_count,
                    "char_count": getattr(s, "char_count", 0),
                    "requests": s.request_count,
                    "processing": s.processing,
                    "evict_score": round(score, 2),
                    "latest_token": s.latest_token,
                    "matched_chars": s.matched_chars,
                    "new_chars": s.new_chars,
                    "source": getattr(s, "source", ""),
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
        f"Proxy started: {args.proxy_host}:{args.proxy_port} -> {args.llama_url}  slots={args.slots}"
    )
    web.run_app(
        app, host=args.proxy_host, port=args.proxy_port, print=None, access_log=None
    )


if __name__ == "__main__":
    main()
