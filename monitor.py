#!/usr/bin/env python3
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

# --------------------------------------------------
# Configuration loading
# --------------------------------------------------
# The project resides in the llama-proxy folder; use a relative path to locate config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
except Exception:
    config = {}

PROXY_HOST = config.get("proxy", {}).get("host", "127.0.0.1")
PROXY_PORT = config.get("proxy", {}).get("port", 8888)
# If proxy listens on 0.0.0.0, use 127.0.0.1 for local queries
PROXY_URL = f"http://{'127.0.0.1' if PROXY_HOST == '0.0.0.0' else PROXY_HOST}:{PROXY_PORT}/proxy/status"

LLAMA_BASE_URL = config.get("llama_server", {}).get("url", "http://10.0.0.20:11400")
LLAMA_URL = f"{LLAMA_BASE_URL}/slots"

SSH_HOST = config.get("llama_server", {}).get("ssh_host", "krsz@10.0.0.20")
LLAMA_LOG_PATH = config.get("llama_server", {}).get("log_path", "/tmp/llama.log")


# --------------------------------------------------
# Global data and utilities
# --------------------------------------------------

state_lock = threading.Lock()
proxy_data = {}
llama_data = {}
prefill_progress = {}
last_task_ids = {}
proxy_logs = deque(maxlen=8)


def human_format(num):
    """Format large numbers into k, M, G units, showing integer-style output."""
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

    val = f"{num:.1f}".rstrip("0").rstrip(".")
    return f"{val}{['', 'k', 'M', 'G', 'T'][magnitude]}"


# --------------------------------------------------
# Background data collection threads
# --------------------------------------------------


def fetch_proxy():
    """Periodically fetch /proxy/status from the local proxy for UI data."""
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
    """Tail the local proxy.log file and keep recent events for display."""
    global proxy_logs
    # Log file path relative to the project directory
    log_file = os.path.join(BASE_DIR, "proxy.log")
    cmd = ["tail", "-n", "0", "-F", log_file]
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
            if "/proxy/status" in line:
                continue
            with state_lock:
                proxy_logs.append(line.strip())
    except Exception:
        pass


def fetch_llama():
    """Periodically query the llama-server /slots endpoint to enrich the UI."""
    global llama_data
    while True:
        try:
            data = requests.get(LLAMA_URL, timeout=1).json()
            with state_lock:
                # convert list to dict keyed by slot id for fast lookup
                llama_data = {slot["id"]: slot for slot in data}
        except Exception:
            pass
        time.sleep(0.5)


def tail_llama_log():
    """
    SSH-tail the remote llama-server log and extract Prefill/progress lines.

    Expected log lines contain phrases like "prompt processing progress" and "prompt processing done".
    The monitor extracts the slot id and progress value and stores them in prefill_progress.
    """
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


# --------------------------------------------------
# UI layout and rendering
# --------------------------------------------------


def generate_layout():
    """Construct the Rich layout used for live monitoring."""
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

    # 1. Header area
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

    # 2. Slot resource allocation table (with row separators)
    res_table = Table(
        expand=True, show_header=True, header_style="bold magenta", show_lines=True
    )
    res_table.add_column("Slot", justify="center", ratio=1)
    res_table.add_column("State", justify="center", ratio=3)
    res_table.add_column("Msgs", justify="center", ratio=1)
    res_table.add_column("Total Chars", justify="center", ratio=4)
    res_table.add_column("Evict Score", justify="center", ratio=2)
    res_table.add_column("Progress / Status", justify="center", ratio=5)

    # 3. Real-time token stream table (includes SessionID and Source)
    token_table = Table(
        expand=True, show_header=True, header_style="bold blue", show_lines=True
    )
    token_table.add_column("Slot", justify="center", ratio=1)
    token_table.add_column("Session", justify="center", ratio=2)
    token_table.add_column("Source", justify="center", ratio=2)
    token_table.add_column("Live Generated Tokens", justify="left", ratio=10)

    slots = p_data.get("slots", [])

    # Precompute the highest evict score among non-processing slots for styling
    max_score = -1.0
    for slot in slots:
        if not slot.get("processing") and slot.get("session_id"):
            score = slot.get("evict_score", 0)
            if score > max_score:
                max_score = score

    for slot in slots:
        s_id = slot["slot_id"]
        is_processing = slot.get("processing", False)

        # Task switch detection and progress cleanup
        l_slot = l_data.get(s_id, {})
        current_task_id = l_slot.get("id_task", -1)
        if last_task_ids.get(s_id) != current_task_id:
            with state_lock:
                prefill_progress.pop(s_id, None)
            last_task_ids[s_id] = current_task_id

        # State text
        if is_processing:
            state = "[bold red]Processing[/bold red]"
        elif slot.get("session_id"):
            state = "[green]Idle (Cached)[/green]"
        else:
            state = "[dim]Empty[/dim]"

        # Evict score display logic
        raw_score = slot.get("evict_score", 0)
        score_val = int(raw_score)
        if is_processing or not slot.get("session_id"):
            evict_score = Text("-", style="dim")
        else:
            # Highlight highest score in red
            style = (
                "bold red" if (max_score > 0 and raw_score >= max_score) else "white"
            )
            evict_score = Text(str(score_val), style=style)

        # Detailed running status (Prefill / Generating)
        op_status = "[dim]-[/dim]"
        n_decoded = 0
        latest_token = slot.get("latest_token", "")
        source = slot.get("source", "")

        if is_processing:
            if "next_token" in l_slot and len(l_slot["next_token"]) > 0:
                n_decoded = l_slot["next_token"][0].get("n_decoded", 0)

            if latest_token and n_decoded > 0:
                op_status = (
                    f"[bold cyan]Generating ({human_format(n_decoded)} T)[/bold cyan]"
                )
            elif source == "TitleGen":
                op_status = "[bold green]Generating Title...[/bold green]"
            else:
                prog = prog_data.get(s_id, -1.0)
                if prog >= 0:
                    op_status = f"[bold yellow]Prefill: {prog * 100:.1f}%[/bold yellow]"
                elif (
                    slot.get("matched_chars", 0) > 0 and slot.get("new_chars", 0) < 1500
                ):
                    op_status = "[bold green]Fast Cache Hit[/bold green]"
                else:
                    op_status = "[bold yellow]Prefill...[/bold yellow]"

        # Character statistics (show matched + new when processing)
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

        # Fill token stream table
        session_id = slot.get("session_id")
        if session_id:
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
                f"{s_id}",
                session_id[:8],
                source[:15],
                token_display,
            )

    # 4. Footer: queue status and recent logs
    waiting = p_data.get("waiting_requests", 0)
    queue_style = "bold yellow" if waiting > 0 else "dim"
    queue_text = Text(f"Waitlist: {waiting}", style=queue_style, justify="center")

    log_text = Text()
    for log_entry in logs:
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
    # Start background threads for fetching proxy and llama data and for tailing logs
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
