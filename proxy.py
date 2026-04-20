#!/usr/bin/env python3
"""
llama-proxy: session → slot 路由代理

配合以下 llama-server 参数使用:
  --parallel 3          # slot 数量，按需调整
  --kv-unified          # 统一 KV 池，slot 间共享 system prompt
  --cache-ram 0         # 禁用内存级别缓存（防止 clear_idle 静默清空 GPU KV）
  --ctx-size 65536      # 总 KV 池大小

原理:
  - 每个 session_id 绑定到一个固定 slot
  - 同一 session 的请求始终发到同一 slot，KV cache 持续累积
  - 使用完整的前 3 条消息作为 Session ID，确保前缀匹配的稳定性
  - Cost-Aware 驱逐策略：综合考虑闲置时间与重算成本（长对话更难被驱逐）
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("llama-proxy")

# ─────────────────────────────────────────────
# 全局配置
# ─────────────────────────────────────────────

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)

PROXY_HOST = _config.get("proxy", {}).get("host", "0.0.0.0")
PROXY_PORT = _config.get("proxy", {}).get("port", 8888)
NUM_SLOTS = _config.get("proxy", {}).get("slots", 4)
LLAMA_URL = _config.get("llama_server", {}).get("url", "http://10.0.0.20:11400")


# ─────────────────────────────────────────────
# 数据结构
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
# 工具函数
# ─────────────────────────────────────────────


def make_session_id(messages: list) -> str:
    """
    根据消息内容生成稳定的 session ID。
    提取前 3 条消息的完整纯文本内容做哈希，避免因截断导致的缓存冲突。
    """
    extracted = []
    for m in messages[:3]:
        role = m.get("role", "")
        content = m.get("content", "")

        text_content = ""
        if isinstance(content, str):
            text_content = content
        elif isinstance(content, list):
            # 处理多模态结构，仅提取文本
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


# ─────────────────────────────────────────────
# 代理核心
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
        log.info(f"初始化: {num_slots} 个 slot")

    @property
    def http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            connector = aiohttp.TCPConnector(limit=20)
            timeout = aiohttp.ClientTimeout(total=None, connect=10)
            self._http = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self._http

    def _get_slot_for_session(self, session_id: str, messages: list) -> tuple[int, str]:
        # 1. 已有绑定
        if session_id in self.sessions:
            slot_id = self.sessions[session_id].slot_id
            self.sessions.move_to_end(session_id)
            return slot_id, "复用已有 slot"

        # 2. 有空闲 slot
        for slot in self.slots.values():
            if slot.is_idle():
                return slot.slot_id, "分配空闲 slot"

        # 3. Cost-Aware (成本感知) 驱逐
        if not self.sessions:
            # 防止状态不一致时（如之前请求失败导致 sessions 为空但 slots 仍被标记占用）崩溃
            return 0, "强制复用 slot 0 (状态异常恢复)"

        now = time.time()
        best_evict_session_id = None
        max_evict_score = -1.0

        for sid, sinfo in self.sessions.items():
            idle_time = now - sinfo.last_used
            # 驱逐得分 = (闲置时间(秒) * 1000) / (总字符数 + 1000)
            # 字符越多（重算成本越高），分母越大，得分越低，越不容易被踢掉
            score = (idle_time * 1000) / (sinfo.char_count + 1000)
            if score > max_evict_score:
                max_evict_score = score
                best_evict_session_id = sid

        evict_session_id = best_evict_session_id or next(iter(self.sessions))
        evict_slot_id = self.sessions[evict_session_id].slot_id

        del self.sessions[evict_session_id]
        self.slots[evict_slot_id].session_id = None  # 防止后续 forward 失败导致状态遗留
        return (
            evict_slot_id,
            f"驱逐 session {evict_session_id[:8]} (分数: {max_evict_score:.1f})",
        )

    def _bind_session(self, session_id: str, slot_id: int, messages: list):
        now = time.time()
        msg_count = len(messages)

        char_count = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                char_count += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        char_count += len(part.get("text", ""))
            else:
                char_count += len(str(content))

        slot = self.slots[slot_id]
        slot.session_id = session_id
        slot.msg_count = msg_count
        slot.char_count = char_count
        slot.last_used = now
        slot.request_count += 1

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

            # 强制启用缓存
            body["cache_prompt"] = True
            session_id = make_session_id(messages)

            # 计算总字符数，用于衡量缓存权重
            total_chars = sum(len(str(m.get("content", ""))) for m in messages)

            async with self.lock:
                slot_id, reason = self._get_slot_for_session(session_id, messages)

                cached_msgs = self.slot_messages.get(slot_id, [])
                prefix_len = common_prefix_len(messages, cached_msgs)

                log.info(
                    f"session={session_id[:8]} → slot={slot_id} | {reason} | "
                    f"msgs={len(messages)} chars={total_chars} 匹配={prefix_len}"
                )

                body["id_slot"] = slot_id

                # 提前绑定 session 并更新状态，立即释放锁，防止流式传输过程长时间阻塞其他并发请求
                self._bind_session(session_id, slot_id, messages)
                self.slot_messages[slot_id] = list(messages)
                self.slots[slot_id].processing = True

            try:
                return await self._forward(request, body, stream)
            except (Exception, asyncio.CancelledError) as e:
                log.info(f"客户端请求中断 ({type(e).__name__})")
                raise
            finally:
                self.slots[slot_id].processing = False
        finally:
            self.semaphore.release()

    async def _forward(self, original_request: web.Request, body: dict, stream: bool):
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
                    async for chunk in upstream.content.iter_any():
                        await resp.write(chunk)
                await resp.write_eof()
            except (Exception, asyncio.CancelledError) as e:
                # 忽略客户端断开引起的写入异常，正常返回 resp 防止 aiohttp 抛出 500
                log.info(f"流传输中断 (客户端已断开): {type(e).__name__}")
            return resp
        else:
            async with self.http.post(
                url, json=body, headers=headers, timeout=timeout
            ) as upstream:
                data = await upstream.read()
                return web.Response(
                    status=upstream.status,
                    headers={"Content-Type": "application/json"},
                    body=data,
                )

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
    """
    [答疑] monitor 是如何捕获远端 llama-server 的输出的？原理是什么？
    原理：monitor.py 在后台启动了一个子线程，使用 subprocess 执行了一条 SSH 命令：
    `ssh krsz@10.0.0.20 tail -n 0 -F /tmp/llama.log`
    这相当于开了一个长连接管道，llama-server 只要打出一行日志，就会通过 SSH 实时推送到本地内存中。
    我们通过正则 (re.search) 在内存中拦截 "prompt processing progress" 这行并提取进度，
    从而完全不侵入 llama-server 的 HTTP 代码，实现了 0 延迟的 Prefill 进度获取。
    """
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
        f"代理启动: {args.proxy_host}:{args.proxy_port} → {args.llama_url}  slots={args.slots}"
    )
    web.run_app(app, host=args.proxy_host, port=args.proxy_port, print=None)


if __name__ == "__main__":
    main()
