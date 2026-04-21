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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")


def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


config = load_config()
PROXY_HOST = config.get("proxy", {}).get("host", "127.0.0.1")
PROXY_PORT = config.get("proxy", {}).get("port", 8888)
PROXY_URL = f"http://{'127.0.0.1' if PROXY_HOST == '0.0.0.0' else PROXY_HOST}:{PROXY_PORT}/proxy/status"

SSH_HOST = config.get("llama_server", {}).get("ssh_host", "")
LOG_PATH = config.get("llama_server", {}).get("log_path", "/tmp/llama.log")

state_lock = threading.Lock()
proxy_data = {}
proxy_logs = deque(maxlen=10)
remote_events = {}
slot_stats = {}
slot_buffers = {}


def human_format(num):
    if num is None:
        return "0"
    try:
        num = float(num)
    except:
        return str(num)
    magnitude = 0
    while abs(num) >= 1000 and magnitude < 4:
        magnitude += 1
        num /= 1000.0
    if magnitude == 0:
        return str(int(num))
    val = f"{num:.1f}".rstrip("0").rstrip(".")
    return f"{val}{['', 'k', 'M', 'G', 'T'][magnitude]}"


def strip_ansi(text):
    return re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])").sub("", text)


def fetch_proxy_loop():
    global proxy_data
    while True:
        try:
            resp = requests.get(PROXY_URL, timeout=1)
            if resp.status_code == 200:
                data = resp.json()
                with state_lock:
                    proxy_data = data
            else:
                with state_lock:
                    proxy_data = {}
        except Exception:
            with state_lock:
                proxy_data = {}
        time.sleep(0.5)


def tail_proxy_log_loop():
    log_file = os.path.join(BASE_DIR, "proxy.log")
    while True:
        if not os.path.exists(log_file):
            time.sleep(2)
            continue
        try:
            cmd = ["tail", "-n", "0", "-F", log_file]
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            for line in iter(p.stdout.readline, ""):
                if not line or "/proxy/status" in line:
                    continue
                with state_lock:
                    proxy_logs.append(line.strip())
            p.wait()
        except Exception:
            time.sleep(2)


def tail_remote_llama_log_loop():
    if not SSH_HOST:
        return
    while True:
        try:
            cmd = ["ssh", SSH_HOST, f"tail -n 0 -F {LOG_PATH}"]
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            for raw_line in iter(p.stdout.readline, ""):
                if not raw_line:
                    break
                line = strip_ansi(raw_line)
                now = time.time()
                m_id = re.search(r"id\s+(\d+)", line)
                if not m_id:
                    continue
                slot_id = int(m_id.group(1))

                if "stopped by EOS" in line or "release:" in line:
                    with state_lock:
                        remote_events.pop(slot_id, None)
                        slot_stats.pop(slot_id, None)
                        slot_buffers.pop(slot_id, None)
                    continue

                m_prog = re.search(r"progress\s*[:=,]\s*([0-9.]+)", line)
                m_tokens = re.search(r"n_tokens\s*[:=,]\s*(\d+)", line)
                m_dec = re.search(r"n_decoded\s*[:=,]\s*(\d+)", line)
                m_rem = re.search(r"n_remaining\s*[:=,]\s*(-?\d+)", line)

                event_text = ""
                if m_prog:
                    prog_val = float(m_prog.group(1))
                    speed_suffix = ""
                    with state_lock:
                        s = slot_stats.get(
                            slot_id, {"last_n": 0, "last_t": now, "total": 0}
                        )
                        if m_tokens:
                            s["total"] = int(m_tokens.group(1))
                        total_tokens = s["total"]
                        current_n = prog_val * total_tokens
                        if now > s["last_t"] and current_n > s["last_n"]:
                            speed = (current_n - s["last_n"]) / (now - s["last_t"])
                            speed_suffix = f" | {speed:.1f} T/s"
                        s.update({"last_n": current_n, "last_t": now})
                        slot_stats[slot_id] = s
                    event_text = f"[bold yellow]Slot {slot_id}: Prefill {prog_val * 100:.1f}% ({total_tokens} T){speed_suffix}[/bold yellow]"
                elif m_dec or m_rem:
                    n_dec_val = int(m_dec.group(1)) if m_dec else 0
                    rem_val = m_rem.group(1) if m_rem else "?"
                    speed_suffix = ""
                    with state_lock:
                        s = slot_stats.get(
                            slot_id, {"last_n": 0, "last_t": now, "total": 0}
                        )
                        if now > s["last_t"] and n_dec_val > s["last_n"]:
                            speed = (n_dec_val - s["last_n"]) / (now - s["last_t"])
                            speed_suffix = f" | {speed:.1f} T/s"
                        s.update({"last_n": n_dec_val, "last_t": now})
                        slot_stats[slot_id] = s
                    if rem_val == "-1":
                        rem_val = "∞"
                    event_text = f"[bold cyan]Slot {slot_id}: Decoding {human_format(n_dec_val)} T (rem: {rem_val}){speed_suffix}[/bold cyan]"

                    m_token = re.search(r"next token:\s*\d+\s*'(.*)'", line)
                    if m_token:
                        t = m_token.group(1).replace("\\n", " ").replace("\\t", " ")
                        with state_lock:
                            slot_buffers[slot_id] = (slot_buffers.get(slot_id, "") + t)[
                                -80:
                            ]

                if event_text:
                    with state_lock:
                        remote_events[slot_id] = {"text": event_text, "time": now}
            p.wait()
        except Exception:
            time.sleep(5)


def generate_layout():
    now = time.time()
    with state_lock:
        p_data = dict(proxy_data)
        logs = list(proxy_logs)
        stale = [sid for sid, data in remote_events.items() if now - data["time"] > 20]
        for sid in stale:
            remote_events.pop(sid)
            slot_stats.pop(sid, None)
            slot_buffers.pop(sid, None)

        r_events = []
        for sid, data in remote_events.items():
            r_events.append(data["text"])
            if sid in slot_buffers:
                r_events.append(f"[dim]  > {slot_buffers[sid]}[/dim]")

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=10),
    )

    proxy_online = bool(p_data)
    header_text = Text.from_markup(
        f"[bold cyan]Llama Proxy Monitor[/bold cyan] | Status: {'[bold green]ONLINE[/bold green]' if proxy_online else '[bold red]OFFLINE[/bold red]'} | [bold cyan]Slots: {p_data.get('num_slots', '?')}[/bold cyan]",
        justify="center",
    )
    layout["header"].update(Panel(header_text))

    res_table = Table(
        expand=True, show_header=True, header_style="bold magenta", show_lines=True
    )
    res_table.add_column("Slot", justify="center", ratio=1)
    res_table.add_column("State", justify="center", ratio=2)
    res_table.add_column("Session", justify="center", ratio=2)
    res_table.add_column("Msgs", justify="center", ratio=1)
    res_table.add_column("Total Chars", justify="center", ratio=2)
    res_table.add_column("Evict Score", justify="center", ratio=2)

    slots = p_data.get("slots", [])
    max_score = max(
        [
            s.get("evict_score", 0)
            for s in slots
            if not s.get("processing") and s.get("session_id")
        ]
        or [-1]
    )

    for slot in slots:
        s_id = slot["slot_id"]
        is_proc = slot.get("processing", False)
        sid = slot.get("session_id")
        state = (
            "[bold red]Processing[/bold red]"
            if is_proc
            else ("[green]Idle (Cached)[/green]" if sid else "[dim]Empty[/dim]")
        )
        raw_score = slot.get("evict_score", 0)
        score_style = (
            "bold red"
            if (not is_proc and sid and raw_score >= max_score > 0)
            else "white"
        )
        evict_score = (
            Text(str(int(raw_score)), style=score_style)
            if (not is_proc and sid)
            else Text("-", style="dim")
        )

        res_table.add_row(
            str(s_id),
            state,
            str(sid[:8]) if sid else "[dim]-[/dim]",
            human_format(slot.get("msg_count", 0)),
            human_format(slot.get("char_count", 0)),
            evict_score,
        )

    waiting = p_data.get("waiting_requests", 0)
    wait_txt = Text.from_markup(
        f"Waitlist: [bold yellow]{waiting}[/bold yellow]"
        if waiting > 0
        else "Waitlist: [dim]0[/dim]"
    )
    main_lay = Layout()
    main_lay.split_column(Layout(res_table, ratio=1), Layout(wait_txt, size=1))
    layout["main"].update(Panel(main_lay, title="Slot Resource Allocation"))

    ev_text = Text()
    for rev in r_events:
        ev_text.append(Text.from_markup(rev + "\n"))
    if r_events:
        ev_text.append("-" * 20 + "\n", style="dim")
    for entry in logs:
        clean = re.sub(r"^\d{2}:\d{2}:\d{2} ", "", entry)
        ev_text.append(
            clean + "\n",
            style="bold red"
            if "[ERROR]" in clean
            else ("yellow" if "[WARNING]" in clean else "dim white"),
        )
    layout["footer"].update(Panel(ev_text, title="Events"))

    return layout


def main():
    threading.Thread(target=fetch_proxy_loop, daemon=True).start()
    threading.Thread(target=tail_proxy_log_loop, daemon=True).start()
    threading.Thread(target=tail_remote_llama_log_loop, daemon=True).start()
    time.sleep(0.5)
    with Live(generate_layout(), refresh_per_second=4, screen=True) as live:
        while True:
            try:
                live.update(generate_layout())
            except Exception:
                pass
            time.sleep(0.25)


if __name__ == "__main__":
    main()
