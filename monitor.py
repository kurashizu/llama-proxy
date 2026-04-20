import os
import re
import subprocess
import threading
import time

import requests
import yaml
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

PROXY_HOST = config.get("proxy", {}).get("host", "127.0.0.1")
PROXY_PORT = config.get("proxy", {}).get("port", 8888)
PROXY_URL = f"http://{'127.0.0.1' if PROXY_HOST == '0.0.0.0' else PROXY_HOST}:{PROXY_PORT}/proxy/status"

LLAMA_BASE_URL = config.get("llama_server", {}).get("url", "http://10.0.0.20:11400")
LLAMA_URL = f"{LLAMA_BASE_URL}/slots"

SSH_HOST = config.get("llama_server", {}).get("ssh_host", "krsz@10.0.0.20")
LLAMA_LOG_PATH = config.get("llama_server", {}).get("log_path", "/tmp/llama.log")

from collections import deque

# 全局状态字典与锁
state_lock = threading.Lock()
proxy_data = {}
llama_data = {}
prefill_progress = {}
last_task_ids = {}  # 记录每个 slot 上一次的任务 ID，用于检测任务切换
proxy_logs = deque(maxlen=8)


def fetch_proxy():
    """后台轮询 Proxy 状态"""
    global proxy_data
    while True:
        try:
            data = requests.get(PROXY_URL, timeout=1).json()
            with state_lock:
                proxy_data = data
        except Exception:
            pass
        time.sleep(0.5)


def tail_proxy_log():
    """后台实时获取 proxy.log 的最新几行"""
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
            with state_lock:
                proxy_logs.append(line.strip())
    except Exception:
        pass


def fetch_llama():
    """后台轮询 llama-server 内部 slots 状态以获取生成 Token 数"""
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
    """后台通过 SSH tail 日志来实时捕获 Prefill 进度"""
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

                # 匹配 Prefill 进度: slot update_slots: id 0 | ... progress = 0.891892
                if "prompt processing progress" in line:
                    m = re.search(r"id\s+(\d+)\s*\|.*progress = (0\.\d+|1\.\d+)", line)
                    if m:
                        slot_id = int(m.group(1))
                        prog = float(m.group(2))
                        with state_lock:
                            prefill_progress[slot_id] = prog

                # 匹配 Prefill 完成
                elif "prompt processing done" in line:
                    m = re.search(r"id\s+(\d+)\s*\|", line)
                    if m:
                        slot_id = int(m.group(1))
                        with state_lock:
                            prefill_progress[slot_id] = 1.0

                # 匹配任务释放 (重置进度)
                elif "stop processing" in line or "release:" in line:
                    m = re.search(r"id\s+(\d+)\s*\|", line)
                    if m:
                        slot_id = int(m.group(1))
                        with state_lock:
                            prefill_progress.pop(slot_id, None)
        except Exception:
            pass
        time.sleep(2)  # SSH 断开后等待 2 秒重试


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

    # 1. 标题区
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

    # 2. 槽位状态表格区
    table = Table(expand=True, show_header=True, header_style="bold magenta")
    table.add_column("Slot ID", justify="center", ratio=1)
    table.add_column("State", justify="center", ratio=3)
    table.add_column("Session ID", justify="center", ratio=2)
    table.add_column("Msgs", justify="center", ratio=1)
    table.add_column("Total Chars", justify="center", ratio=2)
    table.add_column("Evict Score", justify="center", ratio=2)
    table.add_column("Progress / Status", justify="center", ratio=4)

    # Live Token 独立表格
    token_table = Table(expand=True, show_header=True, header_style="bold blue")
    token_table.add_column("Slot", justify="center", ratio=1)
    token_table.add_column("Live Generated Tokens", justify="left", ratio=9)

    slots = p_data.get("slots", [])
    for slot in slots:
        s_id = slot["slot_id"]
        is_processing = slot.get("processing", False)

        # 任务切换检测：如果 llama-server 的任务 ID 变了，立刻重置该槽位的本地进度缓存
        l_slot = l_data.get(s_id, {})
        current_task_id = l_slot.get("id_task", -1)
        if last_task_ids.get(s_id) != current_task_id:
            with state_lock:
                prefill_progress.pop(s_id, None)
            last_task_ids[s_id] = current_task_id

        if is_processing:
            state = "[bold red]Processing (Busy)[/bold red]"
        elif slot.get("session_id"):
            state = "[green]Idle (Cached)[/green]"
        else:
            state = "[dim]Empty[/dim]"

        session = slot.get("session_id")
        if session:
            session = session[:8]
        else:
            session = "[dim]None[/dim]"

        evict_score = str(slot.get("evict_score", 0))
        if is_processing or not slot.get("session_id"):
            evict_score = "[dim]-[/dim]"

        # 结合 llama-server 内部数据的状态展示
        op_status = "[dim]-[/dim]"
        if is_processing:
            l_slot = l_data.get(s_id, {})
            n_decoded = 0
            if "next_token" in l_slot and len(l_slot["next_token"]) > 0:
                n_decoded = l_slot["next_token"][0].get("n_decoded", 0)

            # 状态机：优先判断是否已开始吐字
            if n_decoded > 0:
                op_status = f"[bold cyan]Generating (Tokens: {n_decoded})[/bold cyan]"
            else:
                # 只有 n_decoded 为 0 时才显示 Prefill 进度
                prog = prog_data.get(s_id, -1.0)
                if prog >= 0:
                    op_status = f"[bold yellow]Prefill: {prog * 100:.1f}%[/bold yellow]"
                else:
                    op_status = "[bold yellow]Prefill...[/bold yellow]"

        # 显示匹配字符数 (+ 本次新增字符数)
        total_chars = slot.get("char_count", 0)
        new_chars = slot.get("new_chars", 0)
        if is_processing and new_chars > 0:
            matched = total_chars - new_chars
            char_display = f"{matched} [bold yellow](+{new_chars})[/bold yellow]"
        else:
            char_display = str(total_chars)

        table.add_row(
            str(s_id),
            state,
            session,
            str(slot.get("msg_count", 0)),
            char_display,
            evict_score,
            op_status,
        )

        latest_token = slot.get("latest_token", "")
        if session != "[dim]None[/dim]":
            if latest_token:
                token_display = f"[dim]...[/dim][green]{latest_token}[/green]"
            else:
                token_display = "[italic dim]waiting for tokens...[/italic dim]"
            token_table.add_row(f"[bold cyan]Slot {s_id}[/bold cyan]", token_display)

    waiting = p_data.get("waiting_requests", 0)
    queue_style = "bold yellow" if waiting > 0 else "dim"
    queue_text = Text(f"Queue Waitlist: {waiting}", style=queue_style, justify="center")

    log_text = Text()
    for log_entry in logs:
        if "GET /proxy/status" in log_entry:
            continue
        if "[ERROR]" in log_entry:
            log_text.append(log_entry + "\n", style="bold red")
        elif "[WARNING]" in log_entry:
            log_text.append(log_entry + "\n", style="yellow")
        else:
            log_text.append(log_entry + "\n", style="dim white")

    main_layout = Layout()
    main_layout.split_column(
        Layout(Panel(table, title="Slot Resource Allocation"), ratio=3),
        Layout(Panel(token_table, title="Real-time Token Streams"), ratio=2),
    )
    layout["main"].update(main_layout)

    footer_layout = Layout()
    footer_layout.split_row(
        Layout(Panel(queue_text, title="Queue"), ratio=1),
        Layout(Panel(log_text, title="Recent Proxy Events"), ratio=3),
    )
    layout["footer"].update(footer_layout)
    return layout


def main():
    # 启动后台数据采集线程
    threading.Thread(target=fetch_proxy, daemon=True).start()
    threading.Thread(target=fetch_llama, daemon=True).start()
    threading.Thread(target=tail_llama_log, daemon=True).start()
    threading.Thread(target=tail_proxy_log, daemon=True).start()

    # 等待初始数据加载
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
