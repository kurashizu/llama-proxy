# Llama Proxy — Session → Slot

```mermaid
graph TD
  Agent["Agent / Client"]
  Proxy["Llama Proxy"]
  Slots["Server Slots"]
  LServer["llama-server"]
  Monitor["Monitor (SSH tail)"]

  Agent -->|POST /v1/chat/completions| Proxy
  Proxy -->|compute session_id\n(hash of first 3 messages)| Proxy_note[/"Compute session_id\n(hash of first 3 messages)"/]
  Proxy_note --> Proxy
  Proxy -->|forward to mapped slot (id_slot)| Slots
  Slots -->|internal handling\n(shared KV with --kv-unified)| LServer
  Monitor -->|ssh tail -F /tmp/llama.log| LServer
  LServer -->|log lines (includes prefill progress when -lv 4)| Monitor
  Monitor -->|update UI with Prefill & slot metrics| Proxy
  Proxy -->|response stream / JSON| Agent
```

---

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English

### Overview
A compact session-to-slot proxy that forwards OpenAI-compatible chat completion requests to a llama.cpp-based server (llama-server). The proxy binds a stable session ID to a slot so per-slot KV cache can be reused, reducing recomputation and improving concurrency stability. It also includes a cost-aware eviction policy and a monitor that tails remote llama-server logs to extract Prefill/progress information.

> ⚠️ Important: This project is agent-agnostic. Any client or agent that issues OpenAI-compatible chat completion HTTP requests can use this proxy.

### Key features
- 🔗 Session affinity (session → slot) so the same conversation reuses the same slot.
- 🧠 Cost-aware eviction: prefers evicting short/cheap contexts and preserves long/expensive histories.
- 🔒 Deterministic session ID: hash of the full text of the first three messages (role + content).
- 🛡️ Robust handling of cancellations and client disconnects to avoid slot state corruption.
- 📡 Prefill progress monitoring: monitor SSH-tails llama-server logs and extracts progress (requires verbose server logs).

### Requirements
- Python 3.8+ with these common dependencies: `aiohttp`, `pyyaml`, `rich`, `requests`.
- A running `llama-server` (llama.cpp server) reachable from the proxy.
- For Prefill progress monitoring: start `llama-server` with verbose logs (`-lv 4`) and redirect logs to a file accessible via SSH from the machine running the monitor.
- If you use `--kv-unified` on the server, it is strongly recommended to add `--cache-ram 0` to avoid server memory-level cache silently clearing GPU KV pools (which can cause desyncs).
- Keep proxy `slots` <= llama-server `--parallel`.

### Example configuration (`config.yaml`)
Create or edit `.hermes/llama-proxy/config.yaml` in the project directory:

```yaml
proxy:
  host: "0.0.0.0"
  port: 8888
  slots: 4

llama_server:
  url: "http://10.0.0.20:11400"
  ssh_host: "user@10.0.0.20"
  log_path: "/tmp/llama.log"
```

- `slots`: number of logical slots the proxy manages (must be <= llama-server `--parallel`).
- `ssh_host`: user@host used by the monitor to SSH-tail the `log_path`.
- `log_path`: path on the llama-server host where logs are written.

### Recommended llama-server startup (must include `-lv 4`)
The monitor relies on verbose `llama-server` logs to extract Prefill/progress lines. Example:

```bash
nohup ./llama-server \
  -m /path/to/your_model.gguf \
  --parallel 4 \
  --kv-unified \
  --cache-ram 0 \
  --ctx-size 65536 \
  -lv 4 \
  > /tmp/llama.log 2>&1 &
```

Notes:
- `-lv 4` is required for Prefill/progress lines that the monitor extracts.
- Use `--cache-ram 0` when using `--kv-unified` to avoid unexpected memory-level eviction of GPU KV state.
- Ensure `--parallel` is >= proxy `slots`.

### Run proxy and monitor
From the `llama-proxy` directory:

Start proxy (background helper or direct run):

```bash
# helper script (recommended)
./start_proxy.sh

# or run directly
python3 proxy.py --proxy-host 0.0.0.0 --proxy-port 8888 --llama-url http://10.0.0.20:11400 --slots 4
```

Start monitor (foreground, live UI):

```bash
./start_monitor.sh
```

The monitor shows per-slot status, Prefill progress, live token previews, evict scores, and recent events.

### API endpoints
- `POST /v1/chat/completions` — proxy entry for chat completions (OpenAI-compatible).
- `GET /proxy/status` — JSON summary of proxy and slots (used by the monitor).
- All other endpoints are proxied directly to the underlying llama-server (passthrough).

### Minimal client example (curl)
A streaming chat completion example:

```bash
curl -N -X POST "http://localhost:8888/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Write a short poem about coffee."}
    ],
    "stream": true,
    "cache_prompt": true
  }'
```

Integration notes for agents:
- Point your agent's model endpoint to the proxy URL (e.g., `http://<proxy-host>:8888/v1/chat/completions`) instead of the llama-server URL.
- Keep the OpenAI Chat Completions request shape (messages array).
- Provide a stable `user` field or deterministic system prompt to help session hashing and source detection.

---

<a name="chinese"></a>
## 中文

### 概述
本代理将符合 OpenAI Chat Completion 请求格式的请求转发到基于 llama.cpp 的服务端（llama-server），并实现「会话 → 槽位」的亲和路由：把稳定的会话 ID 绑定到固定槽位，从而复用每个槽位的 KV 缓存，减少重复计算并提升并发稳定性。代理内置成本感知驱逐策略，并提供一个监控（monitor）通过 SSH tail 远端日志来提取 Prefill/进度信息。

> ⚠️ 重要：本项目与具体 agent 无关，任何能发送 OpenAI 兼容 HTTP 请求的客户端或 agent 都可以接入本代理。

### 核心特性
- 🔗 会话亲和（session → slot），相同对话复用同一槽位。
- 🧠 成本感知驱逐：优先驱逐短小/低成本会话，保护长对话及高重算成本历史。
- 🔒 稳定的会话 ID：基于前 3 条消息（role + content）的全文哈希生成。
- 🛡️ 对取消、断连和快速切换通道具备鲁棒性，避免槽位状态错乱。
- 📡 Prefill 监控：通过 SSH tail 远端 llama-server 日志提取进度（需要服务器输出详细日志）。

### 前提要求
- Python 3.8+（常用依赖：`aiohttp`, `pyyaml`, `rich`, `requests`）。
- 已部署并可访问的 `llama-server`。
- 若需 Prefill 进度监控：以 `-lv 4` 启动 `llama-server` 并把日志写到文件；monitor 通过 SSH tail 该日志文件。
- 如果使用 `--kv-unified`，建议同时加上 `--cache-ram 0`，以避免服务器的内存级缓存后台清理 GPU KV 导致不同步。
- proxy 的 `slots` 应 ≤ llama-server 的 `--parallel`。

### 配置示例（`config.yaml`）
放置在 `.hermes/llama-proxy/config.yaml`：

```yaml
proxy:
  host: "0.0.0.0"
  port: 8888
  slots: 4

llama_server:
  url: "http://10.0.0.20:11400"
  ssh_host: "user@10.0.0.20"
  log_path: "/tmp/llama.log"
```

### 推荐的 `llama-server` 启动（必须包含 `-lv 4`）
示例：

```bash
nohup ./llama-server \
  -m /path/to/your_model.gguf \
  --parallel 4 \
  --kv-unified \
  --cache-ram 0 \
  --ctx-size 65536 \
  -lv 4 \
  > /tmp/llama.log 2>&1 &
```

说明：
- `-lv 4` 用于输出 Prefill / progress 信息，monitor 依赖该日志行来显示进度。
- 使用 `--kv-unified` 时建议 `--cache-ram 0`，避免内存层缓存带来的 KV 不一致问题。
- `--parallel` 应 ≥ proxy 的 `slots`。

### 启动代理与监控
在 `llama-proxy` 目录下：

启动代理（后台）：

```bash
./start_proxy.sh
# 或
python3 proxy.py --proxy-host 0.0.0.0 --proxy-port 8888 --llama-url http://10.0.0.20:11400 --slots 4
```

启动监控（前台）：

```bash
./start_monitor.sh
```

监控界面会显示每个槽位的状态、Prefill 进度、生成 token 预览、驱逐分数以及近期事件。

### API 端点
- `POST /v1/chat/completions` — 聊天完成请求入口（OpenAI 兼容）。
- `GET /proxy/status` — 返回 JSON 的槽位与代理状态（供 monitor 使用）。
- 其他路径直接代理到 llama-server（passthrough）。

### 最小客户端示例（curl）
示例（流式）：

```bash
curl -N -X POST "http://localhost:8888/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are an assistant."},
      {"role": "user", "content": "Write a short poem about coffee."}
    ],
    "stream": true,
    "cache_prompt": true
  }'
```

### 常见问题排查
- Prefill 显示 `waiting...` 但模型响应很快：
  - 快速响应可能不会输出详细的 prefill 日志行。检查 `proxy.log` 与远端 `llama.log`，确认是否有可供解析的进度行。
- 意外槽位被驱逐导致上下文丢失：
  - 检查是否使用了 `--cache-ram 0`（当使用 `--kv-unified` 时）。查看 `/proxy/status` 的 `evict_score` 并根据需要增加槽位或调整 server 的 `--parallel`。
- 会话匹配错误（因 system prompt 不同）：
  - 避免在 system prompt 中插入时间戳或临时元数据。使用确定性的 system prompt 或在请求中传入稳定的 `user` 字段。

---
