import os
import re
import subprocess
import threading
import time
from collections import deque

import requests
import yaml
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ─────────────────────────────────────────────
# 配置加载
# ─────────────────────────────────────────────

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
except Exception:
    config = {}

PROXY_HOST = config.get("proxy", {}).get("host", "127.0.0.1")
PROXY_PORT = config.get("proxy", {}).get("port", 8888)
# 如果监听 0.0.0.0，则本地连接 127.0.0.1
PROXY_URL = f"http://{'127.0.0.1' if PROXY_HOST == '0.0.0.0' else PROXY_HOST}:{PROXY_PORT}/proxy/status"

LLAMA_BASE_URL = config.get("llama_server", {}).get("url", "http://10.0.0.20:11400")
LLAMA_URL = f"{LLAMA_BASE_URL}/slots"

SSH_HOST = config.get("llama_server", {}).get("ssh_host", "krsz@10.0.0.20")
LLAMA_LOG_PATH = config.get("llama_server", {}).get("log_path", "/tmp/llama.log")


# ─────────────────────────────────────────────
# 全局数据与工具
# ─────────────────────────────────────────────

state_lock = threading.Lock()
proxy_data = {}
llama_data = {}
prefill_progress = {}
last_task_ids = {}
proxy_logs = deque(maxlen=8)


def human_format(num):
    """格式化大数字为 k, M, G 等单位，始终显示为整数部分"""
    if num is None:
        return "0"
    try:
        num = float(num)
    except (ValueError, TypeError):
        return str(num)

    magnitude = 0
    while abs(num) >= 1000 and magnitude < 4:
        magnitude += 1
        num /= 1000.0

    if magnitude == 0:
        return str(int(num))

    # 按照用户要求，单位化后也尽量保持简洁（取整或1位小数）
    val = f"{num:.1f}".rstrip("0").rstrip(".")
    return f"{val}{['', 'k', 'M', 'G', 'T'][magnitude]}"


# ─────────────────────────────────────────────
# 后台数据采集线程
# ─────────────────────────────────────────────


def fetch_proxy():
    global proxy_data
    while True:
        try:
            data = requests.get(PROXY_URL, timeout=1).json()
            with state_lock:
                proxy_data = data
        except Exception:
            with state_lock:
                proxy_data = {}
        time.sleep(0.5)


def tail_proxy_log():
    global proxy_logs
    cmd = ["tail", "-n", "0", "-F", "proxy.log"]
    try:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        for line in iter(p.stdout.readline, ""):
            if not line:
                break
            # 过滤掉高频的心跳检查日志
            if "/proxy/status" in line:
                continue
            with state_lock:
                proxy_logs.append(line.strip())
    except Exception:
        pass


def fetch_llama():
    global llama_data
    while True:
        try:
            data = requests.get(LLAMA_URL, timeout=1).json()
            with state_lock:
                llama_data = {slot["id"]: slot for slot in data}
        except Exception:
            pass
        time.sleep(0.5)


def tail_llama_log():
    global prefill_progress
    while True:
        cmd = ["ssh", SSH_HOST, f"tail -n 0 -F {LLAMA_LOG_PATH}"]
        try:
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            for line in iter(p.stdout.readline, ""):
                if not line:
                    break
                if "prompt processing progress" in line:
                    m = re.search(r"id\s+(\d+)\s*\|.*progress = (0\.\d+|1\.\d+)", line)
                    if m:
                        slot_id = int(m.group(1))
                        prog = float(m.group(2))
                        with state_lock:
                            prefill_progress[slot_id] = prog
                elif "prompt processing done" in line:
                    m = re.search(r"id\s+(\d+)\s*\|", line)
                    if m:
                        slot_id = int(m.group(1))
                        with state_lock:
                            prefill_progress[slot_id] = 1.0
                elif "stop processing" in line or "release:" in line:
                    m = re.search(r"id\s+(\d+)\s*\|", line)
                    if m:
                        slot_id = int(m.group(1))
                        with state_lock:
                            prefill_progress.pop(slot_id, None)
        except Exception:
            pass
        time.sleep(2)


# ─────────────────────────────────────────────
# UI 布局与渲染
# ─────────────────────────────────────────────


def generate_layout():
    global last_task_ids
    with state_lock:
        p_data = dict(proxy_data)
        l_data = dict(llama_data)
        prog_data = dict(prefill_progress)
        logs = list(proxy_logs)

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=10),
    )

    # 1. 标题区 (显示 Proxy 是否在线)
    proxy_online = bool(p_data)
    proxy_status = (
        "[bold green]ONLINE[/bold green]"
        if proxy_online
        else "[bold red]OFFLINE[/bold red]"
    )
    num_slots = p_data.get("num_slots", "?")
    header_text = Text.from_markup(
        f"[bold cyan]Llama Proxy Monitor[/bold cyan] | Status: {proxy_status} | [bold cyan]Slots: {num_slots}[/bold cyan]",
        justify="center",
    )
    layout["header"].update(Panel(header_text))

    # 2. 槽位状态表格 (资源分配)
    res_table = Table(expand=True, show_header=True, header_style="bold magenta")
    res_table.add_column("Slot", justify="center", ratio=1)
    res_table.add_column("State", justify="center", ratio=3)
    res_table.add_column("Msgs", justify="center", ratio=1)
    res_table.add_column("Total Chars", justify="center", ratio=4)
    res_table.add_column("Evict Score", justify="center", ratio=2)
    res_table.add_column("Progress / Status", justify="center", ratio=5)

    # 3. 实时字流表格 (包含 SessionID 和 Source)
    token_table = Table(expand=True, show_header=True, header_style="bold blue")
    token_table.add_column("Slot", justify="center", ratio=1)
    token_table.add_column("Session", justify="center", ratio=2)
    token_table.add_column("Source", justify="center", ratio=2)
    token_table.add_column("Live Generated Tokens", justify="left", ratio=10)

    slots = p_data.get("slots", [])
    for slot in slots:
        s_id = slot["slot_id"]
        is_processing = slot.get("processing", False)

        # 任务切换检测与进度清理
        l_slot = l_data.get(s_id, {})
        current_task_id = l_slot.get("id_task", -1)
        if last_task_ids.get(s_id) != current_task_id:
            with state_lock:
                prefill_progress.pop(s_id, None)
            last_task_ids[s_id] = current_task_id

        # 状态文案
        if is_processing:
            state = "[bold red]Processing[/bold red]"
        elif slot.get("session_id"):
            state = "[green]Idle (Cached)[/green]"
        else:
            state = "[dim]Empty[/dim]"

        # 被踢分数 (整数显示)
        evict_score = str(int(slot.get("evict_score", 0)))
        if is_processing or not slot.get("session_id"):
            evict_score = "[dim]-[/dim]"

        # 详细运行状态 (Prefill / Generating)
        op_status = "[dim]-[/dim]"
        n_decoded = 0
        latest_token = slot.get("latest_token", "")
        source = slot.get("source", "")

        if is_processing:
            if "next_token" in l_slot and len(l_slot["next_token"]) > 0:
                n_decoded = l_slot["next_token"][0].get("n_decoded", 0)

            # 状态机优先级重构：
            # 1. 只要代理抓到了字，且 llama-server 确认了生成，就是 Generating
            if latest_token and n_decoded > 0:
                op_status = (
                    f"[bold cyan]Generating ({human_format(n_decoded)} T)[/bold cyan]"
                )
            # 2. 如果是 TitleGen 且正在处理，直接显示任务属性（因为通常极快）
            elif source == "TitleGen":
                op_status = "[bold green]Generating Title...[/bold green]"
            # 3. 检查 Prefill 日志进度
            else:
                prog = prog_data.get(s_id, -1.0)
                if prog >= 0:
                    op_status = f"[bold yellow]Prefill: {prog * 100:.1f}%[/bold yellow]"
                elif (
                    slot.get("matched_chars", 0) > 0 and slot.get("new_chars", 0) < 1500
                ):
                    # 高度疑似缓存命中
                    op_status = "[bold green]Fast Cache Hit[/bold green]"
                else:
                    op_status = "[bold yellow]Prefill...[/bold yellow]"

        # 字符统计 (显示命中+新增)
        total_chars = slot.get("char_count", 0)
        new_chars = slot.get("new_chars", 0)
        if is_processing and new_chars > 0:
            matched = total_chars - new_chars
            char_display = f"{human_format(matched)} [bold yellow](+{human_format(new_chars)})[/bold yellow]"
        else:
            char_display = human_format(total_chars)

        res_table.add_row(
            str(s_id),
            state,
            human_format(slot.get("msg_count", 0)),
            char_display,
            evict_score,
            op_status,
        )

        # 填充 Token 流表格
        session_id = slot.get("session_id")
        if session_id:
            # 针对 TitleGen 优化：即便还在 processing 且文字没出来，也给个更积极的提示
            if not latest_token and is_processing:
                if source == "TitleGen":
                    token_display = (
                        "[italic yellow]summarizing title...[/italic yellow]"
                    )
                else:
                    token_display = "[italic dim]waiting for model...[/italic dim]"
            elif latest_token:
                token_display = f"[green]{latest_token}[/green]"
            else:
                token_display = "[dim]-[/dim]"

            token_table.add_row(
                f"[bold cyan]Slot {s_id}[/bold cyan]",
                session_id[:8],
                source[:15],
                token_display,
            )

    # 4. 底部排队与日志
    waiting = p_data.get("waiting_requests", 0)
    queue_style = "bold yellow" if waiting > 0 else "dim"
    queue_text = Text(f"Waitlist: {waiting}", style=queue_style, justify="center")

    log_text = Text()
    for log_entry in logs:
        # 去除日志中的时间戳前缀以便节省空间
        cleaned_log = re.sub(r"^\d{2}:\d{2}:\d{2} ", "", log_entry)
        if "[ERROR]" in cleaned_log:
            log_text.append(cleaned_log + "\n", style="bold red")
        elif "[WARNING]" in cleaned_log:
            log_text.append(cleaned_log + "\n", style="yellow")
        else:
            log_text.append(cleaned_log + "\n", style="dim white")

    main_layout = Layout()
    main_layout.split_column(
        Layout(Panel(res_table, title="Slot Resource Allocation"), ratio=3),
        Layout(Panel(token_table, title="Real-time Token Streams"), ratio=2),
    )
    layout["main"].update(main_layout)

    footer_layout = Layout()
    footer_layout.split_row(
        Layout(Panel(queue_text, title="Queue"), ratio=1),
        Layout(Panel(log_text, title="Recent Events"), ratio=3),
    )
    layout["footer"].update(footer_layout)
    return layout


def main():
    # 启动异步线程
    threading.Thread(target=fetch_proxy, daemon=True).start()
    threading.Thread(target=fetch_llama, daemon=True).start()
    threading.Thread(target=tail_llama_log, daemon=True).start()
    threading.Thread(target=tail_proxy_log, daemon=True).start()

    time.sleep(1)

    with Live(generate_layout(), refresh_per_second=4, screen=True) as live:
        while True:
            try:
                live.update(generate_layout())
            except Exception:
                pass
            time.sleep(0.25)


if __name__ == "__main__":
    main()
