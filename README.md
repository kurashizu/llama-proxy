# Hermes Llama Proxy (Session → Slot)

这是一个专为大语言模型（基于 `llama.cpp server`）打造的高性能、防阻塞、带重算成本感知（Cost-Aware）的对话并发管理与缓存路由代理层。

完美兼容 `Hermes Agent` 和 `Qwen 3.6` 等具备复杂思考逻辑及循环架构（Recurrent）的模型。

---

## 核心特性

1. **防并发压垮 (Smart Queueing)**：通过异步信号量（Semaphore），允许超过槽位数量的请求平滑排队，拒绝暴力并发导致 `llama-server` 崩溃或缓存疯狂换出（Thrashing）。
2. **算力成本感知 (Cost-Aware LRU)**：相比于“谁最久没说话踢掉谁”的纯 LRU，我们引入了总字符数计分公式。系统宁可献祭闲置了 1 分钟的“2句话短对话”，也会誓死保卫你那闲置了 1 小时的“3万字超长上下文”不被踢掉。
3. **完美前缀匹配**：不再截断 500 个字符，而是提取前 3 条消息的完整文本进行精准 Hash，支持多模态图片的自动文本过滤。
4. **强悍的异常兜底**：全面接住因为用户切频道、UI 刷新、网络闪断导致的 HTTP 连接断开问题，保障代理进程 `proxy.py` 的槽位状态字典永不脱轨。
5. **0 延迟预填充 (Prefill) 监控**：通过纯内存 SSH 管道直连远端日志，完美捕捉大模型极其缓慢的提示词吞吐阶段，提供 `htop` 级的全屏数据流监控。

---

## 安装与配置

所有关键配置均统一收口于本目录下的 `config.yaml` 文件中。无需在脚本中硬编码任何参数。

### `config.yaml` 示例配置
```yaml
proxy:
  host: "0.0.0.0"      # 代理监听地址
  port: 8888           # 对外暴露给客户端 (Hermes Agent) 的端口
  slots: 4             # 最大并发槽位数 (必须与 llama-server 的 --parallel 保持一致或更小)

llama_server:
  url: "http://10.0.0.20:11400"     # 运行着 llama.cpp server 的远端机器地址和端口
  ssh_host: "krsz@10.0.0.20"        # SSH 登录远端的账户和 IP
  log_path: "/tmp/llama.log"        # llama-server 的实时日志输出文件路径
```

### llama-server 启动要求
为了使代理正常工作并能让 Monitor 捕获预填充进度，**你必须在启动 `llama-server` 时满足以下两个条件：**
1. **添加 `-lv 4` 参数**：开启详细日志，以便打印出底层的进度信息。
2. **重定向标准输出到日志文件**：必须将输出写入 `config.yaml` 中配置的 `log_path`。
例如（推荐使用 nohup 后台运行）：
```bash
nohup ./llama-server -m your_model.gguf ... -lv 4 > /tmp/llama.log 2>&1 &
```

---

## 如何启动与使用

### 1. 启动 Proxy 网关 (后台运行)
直接运行：
```bash
./start_proxy.sh
```
这将在后台拉起 `proxy.py`，它会自动读取 `config.yaml` 的配置，挂载在 `8888` 端口（或你在配置文件里指定的端口）。产生的日志会自动记录在当前目录的 `proxy.log` 中。

### 2. 启动可视化控制台 (前台运行)
当你想要查看实时吞吐量时，直接运行：
```bash
./start_monitor.sh
```
这会呼出一个极具科技感的全屏 UI 面板。

---
