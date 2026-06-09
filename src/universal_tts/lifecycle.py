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


def port_pid(port: int | None) -> str | None:
    if not port:
        return None
    return run(f"lsof -tiTCP:{int(port)} -sTCP:LISTEN | head -1", 5)[1] or None


class ProcessLifecycle:
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

    def start(self) -> dict:
        if self.launchd_label:
            if self.plist and Path(self.plist).exists():
                run(f"launchctl bootstrap {user_domain()} {shell_quote(self.plist)}", 20)
            rc, out = run(f"launchctl kickstart -k {user_domain()}/{self.launchd_label}", 20)
            return {"method": "launchd", "exit_code": rc, "output": out}
        if self.command:
            existing = port_pid(self.port)
            if existing:
                return {"method": "command", "already_listening": True, "pid": existing}
            log_path = self.log_dir / f"{self.provider_id}.log"
            log = open(log_path, "ab", buffering=0)
            proc = subprocess.Popen(self.command, shell=True, cwd=self.cwd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
            self.pid = proc.pid
            return {"method": "command", "pid": proc.pid, "log": str(log_path)}
        return {"method": "inline", "started_at": time.time()}

    def stop(self) -> dict:
        if self.launchd_label:
            rc, out = run(f"launchctl bootout {user_domain()}/{self.launchd_label}", 20)
            return {"method": "launchd", "exit_code": rc, "output": out}
        killed: list[str] = []
        if self.pid:
            try:
                os.killpg(int(self.pid), signal.SIGTERM)
                killed.append(str(self.pid))
            except Exception:
                pass
        if self.port:
            for pid in run(f"lsof -tiTCP:{int(self.port)} -sTCP:LISTEN", 5)[1].splitlines():
                if pid and pid not in killed:
                    run(f"kill {int(pid)}", 5)
                    killed.append(pid)
        return {"method": "command", "killed": killed}
