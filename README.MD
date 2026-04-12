# LlamaClaw-CacheProxy

[English](#english) | [中文说明](#中文说明)

---

## English

A lightweight CacheProxy between [OpenClaw](https://openclaw.dev) and [llama.cpp](https://github.com/ggml-org/llama.cpp) (llama-server) that solves high TTFT (Time To First Token) issues caused by KV cache eviction.

When using multiple channels (WebChat, Discord, Telegram, etc.), switching between them normally causes llama-server to evict the KV cache, resulting in massive prefill times. This proxy solves the problem by creating multiple cache slots in llama-server and routing traffic from different channels to their dedicated slots.

### Features
- **Slot Routing**: Routes WebChat, Discord, and background sessions to dedicated KV cache slots.
- **Cache Hit Maximization**: Eliminates long prefill times when switching channels by keeping each channel's context in memory.

### How It Works

```
OpenClaw → [CacheProxy :8888] → llama-server :11400
```

For each incoming request, the proxy:
1. Inspects the system prompt's `## Inbound Context` block to extract the `channel` type.
2. Injects `id_slot` into the request body to pin the session to its dedicated slot.
3. Transparently forwards the request to llama-server with full streaming support.

### Slot Assignment

| Channel | Slot | Notes |
|---------|------|-------|
| Default | 0 | Fallback for cron jobs and subagents |
| Discord | 1 | Detected via `"channel"` containing `discord` |
| WebChat | 2 | Detected via `"channel"` containing `webchat` |

### Requirements

- Python 3.8+
- `flask`
- `requests`
- llama.cpp server running locally or on the network
- OpenClaw with a provider pointing to this proxy

### Installation & Usage

```bash
pip install flask requests
python proxy.py
```

The proxy listens on `0.0.0.0:8888` by default and forwards to `http://10.0.0.20:11400`. Edit `proxy.py` to change these defaults.

### Example Configurations

#### 1. llama-server Start Script

Ensure you allocate enough context size for all slots (`--ctx-size`), and set `--parallel` to match the total number of slots.

```bash
#!/bin/bash

export HF="unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL"
mkdir -p /tmp/llama-slots

./build/bin/llama-server \
    -hf "$HF" \
    -ngl 9999 \
    --ctx-size 196608 \
    -ctk q8_0 \
    -ctv q8_0 \
    --batch-size 5120 \
    --ubatch-size 128 \
    --flash-attn on \
    --slot-save-path /tmp/llama-slots/ \
    --threads 8 \
    --threads-batch 12 \
    -ncmoe 25 \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --alias "llama_cpp_model" \
    --parallel 3 \
    --host "0.0.0.0" \
    --port 11400 \
    --reasoning auto \
    --reasoning-budget 256 \
    --reasoning-budget-message " thinking budget exceeded, let's answer now." \
    --swa-full
```

#### 2. OpenClaw Configuration (`openclaw.json`)

Point your `llama_cpp` provider base URL to the CacheProxy instead of llama-server.

```json
{
    "models": {
        "providers": {
            "llama_cpp": {
                "baseUrl": "http://127.0.0.1:8888/v1",
                "api": "openai-completions",
                "apiKey": "sk-1234",
                "models": [
                    {
                        "id": "default",
                        "name": "default (Custom Provider)",
                        "contextWindow": 65536,
                        "maxTokens": 16384,
                        "input": ["text", "image"],
                        "reasoning": true
                    }
                ]
            }
        }
    },
    "channels": {
        "discord": {
            "enabled": true,
            "token": "YOUR_DISCORD_BOT_TOKEN_HERE"
        }
    }
}
```

### Verifying Slot Assignments

```bash
curl http://10.0.0.20:11400/slots | python3 -m json.tool
```

Each slot should show a distinct `n_past` value corresponding to its session history.

### Known Limitations
- **Slot detection**: Detection is based on the `"channel"` field inside the `## Inbound Context` block. If OpenClaw changes its inbound meta format, the regex pattern in `proxy.py` may need updating.

---

## 中文说明

这是一个介于 [OpenClaw](https://openclaw.dev) 和 [llama.cpp](https://github.com/ggml-org/llama.cpp) (llama-server) 之间的 CacheProxy（缓存代理），专门用于解决切换不同渠道（Channel）时导致缓存被驱逐（Eviction），从而引起每次都需要大量时间重新 Prefill、TTFT (首字生成时间) 过长的问题。

通过在 llama-server 内部创建多个缓存 Slot，代理会根据请求来源的 Channel (如 WebChat, Discord, Telegram 等) 将流量分别路由到不同的 Slot，以达到完美命中缓存的目的。

### 核心功能
- **Slot 路由绑定**：将 Discord、WebChat 以及后台会话路由到专门的 KV 缓存 Slot，防止在不同频道间切换时互相驱逐缓存。
- **最大化缓存命中率**：保留各个 Channel 的对话上下文在显存中，避免跨渠道切换导致的时间开销。

### 工作原理

```
OpenClaw → [CacheProxy :8888] → llama-server :11400
```

对于每个传入的请求，代理会执行以下操作：
1. 检查 system prompt 中的 `## Inbound Context`，提取出 `channel` 字段信息（`discord`、`webchat` 等）。
2. 在请求体中注入 `id_slot` 参数，将请求分配到该频道对应的专属 Slot。
3. 支持流式传输（Streaming）并将请求透明转发给后端的 llama-server。

### Slot 分配策略

| 来源 | Slot ID | 说明 |
|--------|------|-------|
| 默认 / 后台 | 0 | 默认后备选项（未匹配频道或其它后台调用） |
| Discord | 1 | 匹配 `"channel"` 字段中包含 `discord` |
| WebChat | 2 | 匹配 `"channel"` 字段中包含 `webchat` |

### 环境依赖

- Python 3.8+
- `flask`
- `requests`
- 运行中（本地或局域网）的 llama.cpp 服务端
- 将 Provider 接口指向此代理的 OpenClaw

### 安装与运行

```bash
pip install flask requests
python proxy.py
```

代理默认监听端口 `0.0.0.0:8888`，并将请求转发到 `http://10.0.0.20:11400`。可直接编辑 `proxy.py` 顶部的变量进行参数修改。

### 配置示例

#### 1. llama-server 启动脚本示例

请确保通过 `--ctx-size` 给所有的 Slot 分配足够的总上下文大小，并将 `--parallel` 数量设置为何 Slot 总数一致。

```bash
#!/bin/bash

export HF="unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL"
mkdir -p /tmp/llama-slots

./build/bin/llama-server \
    -hf "$HF" \
    -ngl 9999 \
    --ctx-size 196608 \
    -ctk q8_0 \
    -ctv q8_0 \
    --batch-size 5120 \
    --ubatch-size 128 \
    --flash-attn on \
    --slot-save-path /tmp/llama-slots/ \
    --threads 8 \
    --threads-batch 12 \
    -ncmoe 25 \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --alias "llama_cpp_model" \
    --parallel 3 \
    --host "0.0.0.0" \
    --port 11400 \
    --reasoning auto \
    --reasoning-budget 256 \
    --reasoning-budget-message " thinking budget exceeded, let's answer now." \
    --swa-full
```

#### 2. OpenClaw 配置示例 (`openclaw.json`)

在 OpenClaw 的 Provider 配置中，将 `baseUrl` 指向该 CacheProxy 代理服务，而不是直接指向 llama-server。

```json
{
    "models": {
        "providers": {
            "llama_cpp": {
                "baseUrl": "http://127.0.0.1:8888/v1",
                "api": "openai-completions",
                "apiKey": "sk-1234",
                "models": [
                    {
                        "id": "default",
                        "name": "default (Custom Provider)",
                        "contextWindow": 65536,
                        "maxTokens": 16384,
                        "input": ["text", "image"],
                        "reasoning": true
                    }
                ]
            }
        }
    },
    "channels": {
        "discord": {
            "enabled": true,
            "token": "YOUR_DISCORD_BOT_TOKEN_HERE"
        }
    }
}
```

### 验证 Slot 分配

你可以通过请求 llama-server 的 slots 接口来验证会话分配状态：

```bash
curl http://10.0.0.20:11400/slots | python3 -m json.tool
```

如果分配成功，每个 Slot 都会显示与其会话历史相对应的独立的 `n_past` 值。

### 已知限制
- **Slot 匹配检测**：目前检测逻辑完全依赖通过正则提取 `## Inbound Context` 块中的 `"channel"` 字段。如果 OpenClaw 更改了其格式，可能需要更新 `proxy.py` 中的正则表达式。
