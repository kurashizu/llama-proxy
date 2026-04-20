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

# 全局状态字典与锁
state_lock = threading.Lock()
proxy_data = {}
llama_data = {}
prefill_progress = {}


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
    with state_lock:
        p_data = dict(proxy_data)
        l_data = dict(llama_data)
        prog_data = dict(prefill_progress)

    layout = Layout()
    layout.split_column(Layout(name="header", size=3), Layout(name="main"))

    # 1. 标题区
    num_slots = p_data.get("num_slots", "?")
    header_text = Text(
        f"Llama Proxy Monitor | Slots: {num_slots}", justify="center", style="bold cyan"
    )
    layout["header"].update(Panel(header_text))

    # 2. 表格区 (自适应列宽，避免拥挤)
    table = Table(expand=True, show_header=True, header_style="bold magenta")
    table.add_column("Slot ID", justify="center", ratio=1)
    table.add_column("State", justify="center", ratio=4)
    table.add_column("Session ID", justify="center", ratio=1)
    table.add_column("Msgs", justify="center", ratio=1)
    table.add_column("Total Chars", justify="center", ratio=2)
    table.add_column("Evict Score", justify="center", ratio=2)
    table.add_column("Progress / Status", justify="center", ratio=4)

    slots = p_data.get("slots", [])
    for slot in slots:
        s_id = slot["slot_id"]
        is_processing = slot.get("processing", False)

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

            # n_decoded > 0 意味着 Prefill 已结束，正在吐字
            if n_decoded > 0:
                op_status = f"[bold cyan]Generating (Tokens: {n_decoded})[/bold cyan]"
            else:
                # 否则正在进行耗时的 Prefill，显示读取日志得到的进度
                prog = prog_data.get(s_id, -1.0)
                if prog >= 0:
                    op_status = f"[bold yellow]Prefill: {prog * 100:.1f}%[/bold yellow]"
                else:
                    op_status = "[bold yellow]Prefill...[/bold yellow]"

        table.add_row(
            str(s_id),
            state,
            session,
            str(slot.get("msg_count", 0)),
            str(slot.get("char_count", 0)),
            evict_score,
            op_status,
        )

    # 3. 队列区
    waiting = p_data.get("waiting_requests", 0)
    queue_style = "bold yellow" if waiting > 0 else "dim"
    queue_text = Text(
        f"Waiting Requests in Queue: {waiting}", style=queue_style, justify="center"
    )
    queue_panel = Panel(queue_text, title="Queue")

    # 组合布局
    main_layout = Layout()
    main_layout.split_column(
        Layout(Panel(table, title="Slot Occupancy & Performance"), ratio=2),
        Layout(queue_panel, size=3),
    )
    layout["main"].update(main_layout)
    return layout


def main():
    # 启动后台数据采集线程
    threading.Thread(target=fetch_proxy, daemon=True).start()
    threading.Thread(target=fetch_llama, daemon=True).start()
    threading.Thread(target=tail_llama_log, daemon=True).start()

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
