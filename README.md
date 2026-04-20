.hermes/llama-proxy/README.md#L1-300
# Llama Proxy

[English](#english) | [中文](#chinese)

---

## English

### Overview
This proxy routes OpenAI-compatible chat completion requests to a llama.cpp-based server (llama-server) and implements session affinity: a stable session ID is bound to a fixed slot so per-slot KV cache can be re-used and recomputation is minimized. The proxy also implements a cost-aware eviction policy and provides a monitor that reads remote logs to display Prefill/progress information in near real-time.

Important: This project is NOT limited to Hermes Agent. Any agent or client that uses the OpenAI Chat Completions API shape can integrate with this proxy.

### Key features
- Session affinity (session → slot) for persistent KV reuse.
- Cost-aware eviction that prefers evicting short/cheap contexts rather than long/expensive ones.
- Stable session ID generation using the full text of the first three messages to avoid collisions from naive truncation.
- Robust handling of cancellations, client disconnects, and rapid channel switching to avoid slot state corruption.
- Prefill progress monitoring via SSH tailing of remote llama-server logs (requires verbose server logs).

### Requirements
- A running llama-server (llama.cpp server) with a parallel slot configuration.
- For reliable Prefill monitoring, start llama-server with verbose logs (`-lv 4`) and write logs to a stable file accessible by the monitor via SSH.
- When using `--kv-unified`, we recommend also setting `--cache-ram 0` to avoid the server's memory-level cache clearing GPU KV silently (which may cause desyncs).
- Keep proxy `slots` <= llama-server `--parallel`.

### Example config (config.yaml)
Use your local `config.yaml` in the `llama-proxy` folder. Example values:

<pre><code class="language-yaml">
proxy:
  host: "0.0.0.0"
  port: 8888
  slots: 4

llama_server:
  url: "http://10.0.0.20:11400"
  ssh_host: "user@10.0.0.20"
  log_path: "/tmp/llama.log"
</code></pre>

### Recommended llama-server startup (must include `-lv 4`)
Start your llama-server with verbose logs and appropriate flags. Example:

<pre><code>
nohup ./llama-server \
  -m /path/to/your_model.gguf \
  --parallel 4 \
  --kv-unified \
  --cache-ram 0 \
  --ctx-size 65536 \
  -lv 4 \
  > /tmp/llama.log 2>&1 &
</code></pre>

Notes:
- `-lv 4` is required for the monitor to capture Prefill/progress lines.
- Use `--cache-ram 0` with `--kv-unified` for predictable KV behavior across slots.
- `--parallel` should be >= the number of slots used by the proxy; the proxy `slots` value must be <= `--parallel`.

### How to run the proxy and monitor
Start the proxy (background):

<pre><code>
# from the llama-proxy directory
./start_proxy.sh
# or directly
python3 proxy.py --proxy-host 0.0.0.0 --proxy-port 8888 --llama-url http://10.0.0.20:11400 --slots 4
</code></pre>

Start the monitor (foreground):

<pre><code>
./start_monitor.sh
</code></pre>

The monitor opens a live UI that shows per-slot status, Prefill progress, live token previews, evict scores, and recent events.

### API endpoints
- `POST /v1/chat/completions` — proxy entry for chat completions (OpenAI-compatible)
- `GET /proxy/status` — returns JSON with proxy/slot status for the monitor
- All other paths are proxied to the underlying llama-server

### Minimal client example (curl)
Send a streaming chat completion request to the proxy:

<pre><code>
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
</code></pre>

Integration note:
- Any agent that can call HTTP endpoints may use this proxy. Point the agent's model endpoint to the proxy's `/v1/chat/completions`.
- Ensure the agent sends a stable `user` field or deterministic system prompt to keep session hashing consistent.

### Architecture (mermaid)
The monitor tails the remote llama-server log via SSH and extracts Prefill lines to show progress.

<pre><code class="language-mermaid">
sequenceDiagram
  participant Agent as Agent / Client
  participant Proxy as Llama Proxy
  participant Slots as Llama Server Slots
  participant LServer as llama-server
  participant Monitor as Monitor (SSH tail)

  Agent->>Proxy: HTTP POST /v1/chat/completions
  Proxy->>Proxy: compute session_id (hash of first 3 msgs)
  Proxy->>Slots: forward to mapped slot (id_slot)
  Slots->>LServer: internal handling (shared KV with --kv-unified)
  Monitor->>LServer: ssh tail -F /tmp/llama.log
  LServer->>Monitor: log lines (includes prefill progress with -lv 4)
  Monitor->>Proxy: updates UI with Prefill & slot metrics
</code></pre>

### Troubleshooting
- Prefill shows "waiting..." while model responds quickly:
  - Fast responses sometimes skip verbose prefill log lines. Inspect `proxy.log` and `llama.log` for parsing clues.
- Unexpected slot eviction / context loss:
  - Verify `--cache-ram 0` with `--kv-unified`. Inspect `/proxy/status` to see `evict_score`. Increase slots or adjust eviction parameters if necessary.
- Session mismatch due to different system prompts:
  - Avoid ephemeral metadata or timestamps in system prompts. Use deterministic system prompts or set a stable `user` field.

---

## 中文

### 概要
本代理将 OpenAI 兼容的 chat completion 请求路由到基于 llama.cpp 的服务端（llama-server），并实现会话亲和（session → slot）：把稳定的会话 ID 绑定到固定槽位，从而复用每个槽位的 KV 缓存，减少重算。代理包含成本感知驱逐策略，并提供通过 SSH tail 读取远端日志以接近实时显示 Prefill/进度的监控功能。

重要说明：本项目不限于 Hermes Agent。任何遵循 OpenAI Chat Completions API 的客户端/agent 均可接入该代理。

### 关键特性
- 会话亲和：相同会话始终走同一槽位以复用 KV。
- 成本感知驱逐：优先驱逐短小低成本会话，保护长会话。
- 稳定的会话 ID：基于前 3 条消息全文生成哈希以避免截断冲突。
- 可恢复性：对取消、断连和快速切换通道的鲁棒处理，避免槽位状态脱轨。
- Prefill 监控：通过 SSH tail 远端日志（需要服务器 verbose 日志）提取进度。

### 前提要求
- 已部署的 llama-server，并根据需要配置并行槽位。
- 为了可靠捕获 Prefill 进度，需要以详细日志等级启动 llama-server（`-lv 4`）并将日志写入文件，monitor 通过 SSH 读取该日志。
- 若使用 `--kv-unified`，建议同时使用 `--cache-ram 0`，避免服务器的内存级缓存在后台清理 GPU KV 导致不同步。
- proxy 配置的 `slots` 应 ≤ llama-server 的 `--parallel`。

### 示例配置 (config.yaml)
<pre><code class="language-yaml">
proxy:
  host: "0.0.0.0"
  port: 8888
  slots: 4

llama_server:
  url: "http://10.0.0.20:11400"
  ssh_host: "user@10.0.0.20"
  log_path: "/tmp/llama.log"
</code></pre>

### 推荐的 llama-server 启动（必须包含 `-lv 4`）
<pre><code>
nohup ./llama-server \
  -m /path/to/your_model.gguf \
  --parallel 4 \
  --kv-unified \
  --cache-ram 0 \
  --ctx-size 65536 \
  -lv 4 \
  > /tmp/llama.log 2>&1 &
</code></pre>

### 启动代理与监控
启动代理（后台）：

<pre><code>
./start_proxy.sh
# 或
python3 proxy.py --proxy-host 0.0.0.0 --proxy-port 8888 --llama-url http://10.0.0.20:11400 --slots 4
</code></pre>

启动监控（前台）：

<pre><code>
./start_monitor.sh
</code></pre>

### API 端点
- `POST /v1/chat/completions` — 聊天完成请求（OpenAI 兼容）
- `GET /proxy/status` — 返回 JSON 格式的槽位与代理状态（供 monitor 使用）
- 其他路径直接代理到 llama-server

### 最小客户端示例 (curl)
<pre><code>
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
</code></pre>

### 架构图 (mermaid)
<pre><code class="language-mermaid">
sequenceDiagram
  participant Agent as Agent / Client
  participant Proxy as Llama Proxy
  participant Slots as Llama Server Slots
  participant LServer as llama-server
  participant Monitor as Monitor (SSH tail)

  Agent->>Proxy: HTTP POST /v1/chat/completions
  Proxy->>Proxy: compute session_id (hash of first 3 msgs)
  Proxy->>Slots: forward to mapped slot (id_slot)
  Slots->>LServer: internal handling (shared KV with --kv-unified)
  Monitor->>LServer: ssh tail -F /tmp/llama.log
  LServer->>Monitor: log lines (includes prefill progress with -lv 4)
  Monitor->>Proxy: updates UI with Prefill & slot metrics
</code></pre>

### 故障排查要点
- 当 Prefill 显示 `waiting...` 但模型返回很快：
  - 快速响应可能跳过详细 prefill 日志行。查看 `proxy.log` 与 `llama.log` 以排查解析问题。
- 出现意外槽位被驱逐导致上下文丢失：
  - 检查 `--cache-ram 0` 与 `--kv-unified` 是否正确使用，并查看 `/proxy/status` 中的 `evict_score` 值。
- 会话匹配错误（system prompt 不同导致）：
  - 避免在 system prompt 中插入时间戳或临时元数据。使用确定性的 system prompt 或稳定的 `user` 字段。

---
