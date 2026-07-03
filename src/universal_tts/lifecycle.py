from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path


def run(cmd: str, timeout: int = 20) -> tuple[int, str]:
    p = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode, p.stdout.strip()


def user_domain() -> str:
    return f"gui/{os.getuid()}"


def shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def port_pids(port: int | None) -> list[int]:
    if not port:
        return []
    out = run(f"lsof -tiTCP:{int(port)} -sTCP:LISTEN", 5)[1]
    pids: list[int] = []
    for line in out.splitlines():
        try:
            pids.append(int(line.strip()))
        except Exception:
            pass
    return pids


def port_pid(port: int | None) -> str | None:
    pids = port_pids(port)
    return str(pids[0]) if pids else None


class ProcessLifecycle:
    """Own a command-backed provider sidecar.

    Universal TTS is the public service, so command-backed provider listeners are
    treated as sidecars: existing listeners on configured ports are either
    adopted or replaced, children are killed on unload/shutdown, and callers can
    force-kill a wedged listener before respawning it.
    """

    def __init__(self, *, provider_id: str, launchd_label: str | None = None, plist: str | None = None, command: str | None = None, cwd: str | None = None, port: int | None = None, log_dir: str | Path = "logs"):
        self.provider_id = provider_id
        self.launchd_label = launchd_label
        self.plist = plist
        self.command = command
        self.cwd = cwd
        self.port = port
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.pid: int | None = None
        self.process: subprocess.Popen | None = None
        self.adopted_pids: set[int] = set()
        self.started_at: float | None = None
        self.last_start_result: dict | None = None

    def start(self, *, port_conflict_policy: str = "adopt") -> dict:
        if self.launchd_label:
            if self.plist and Path(self.plist).exists():
                run(f"launchctl bootstrap {user_domain()} {shell_quote(self.plist)}", 20)
            rc, out = run(f"launchctl kickstart -k {user_domain()}/{self.launchd_label}", 20)
            self.started_at = time.time()
            self.last_start_result = {"method": "launchd", "exit_code": rc, "output": out}
            return self.last_start_result
        if self.command:
            existing = port_pids(self.port)
            if existing:
                if port_conflict_policy == "replace":
                    self.terminate(force=True, include_port=True)
                    deadline = time.time() + 10
                    while port_pids(self.port) and time.time() < deadline:
                        time.sleep(0.2)
                else:
                    self.adopted_pids.update(existing)
                    self.pid = existing[0]
                    self.started_at = time.time()
                    self.last_start_result = {"method": "command", "adopted": True, "pids": [str(p) for p in existing]}
                    return self.last_start_result
            log_path = self.log_dir / f"{self.provider_id}.log"
            log = open(log_path, "ab", buffering=0)
            proc = subprocess.Popen(self.command, shell=True, cwd=self.cwd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
            self.process = proc
            self.pid = proc.pid
            self.started_at = time.time()
            self.last_start_result = {"method": "command", "pid": proc.pid, "log": str(log_path)}
            return self.last_start_result
        self.started_at = time.time()
        self.last_start_result = {"method": "inline", "started_at": self.started_at}
        return self.last_start_result

    def is_running(self) -> bool:
        if self.launchd_label:
            rc, _out = run(f"launchctl print {user_domain()}/{self.launchd_label} >/dev/null", 5)
            return rc == 0
        if self.process is not None and self.process.poll() is None:
            return True
        if self.port and port_pids(self.port):
            return True
        return False

    def terminate(self, *, force: bool = False, include_port: bool = True) -> dict:
        killed: list[str] = []
        sig = signal.SIGKILL if force else signal.SIGTERM
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(int(self.process.pid), sig)
                killed.append(str(self.process.pid))
            except Exception:
                pass
        for pid in list(self.adopted_pids):
            try:
                os.kill(int(pid), sig)
                killed.append(str(pid))
            except Exception:
                pass
        if include_port and self.port:
            for pid in port_pids(self.port):
                if str(pid) not in killed:
                    try:
                        os.kill(int(pid), sig)
                        killed.append(str(pid))
                    except Exception:
                        pass
        self.process = None
        self.pid = None
        self.adopted_pids.clear()
        return {"method": "command", "signal": int(sig), "killed": killed}

    def stop(self) -> dict:
        if self.launchd_label:
            rc, out = run(f"launchctl bootout {user_domain()}/{self.launchd_label}", 20)
            return {"method": "launchd", "exit_code": rc, "output": out}
        return self.terminate(force=False, include_port=True)

    def kill(self) -> dict:
        if self.launchd_label:
            pid = port_pid(self.port)
            if pid:
                run(f"kill -9 {int(pid)}", 5)
                return {"method": "launchd", "killed": [pid], "signal": 9}
            return {"method": "launchd", "killed": [], "signal": 9}
        return self.terminate(force=True, include_port=True)
