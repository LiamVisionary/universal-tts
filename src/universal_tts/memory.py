from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class MemorySnapshot:
    total_gb: float
    available_gb: float
    available_plus_reclaimable_gb: float


class MemoryGuard:
    def __init__(self, reserve_gb: float):
        self.reserve_gb = float(reserve_gb)

    def assert_can_load(self, provider_id: str, estimate_gb: float, snapshot: MemorySnapshot, force: bool = False) -> dict:
        required = float(estimate_gb) + self.reserve_gb
        available = snapshot.available_plus_reclaimable_gb
        decision = {
            "provider_id": provider_id,
            "estimate_gb": float(estimate_gb),
            "reserve_gb": self.reserve_gb,
            "required_gb": required,
            "available_plus_reclaimable_gb": available,
            "forced": bool(force),
        }
        if available < required and not force:
            raise RuntimeError(f"memory guard refused {provider_id}: needs {required:.2f} GB incl reserve, has {available:.2f} GB")
        return decision


def _run(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10).stdout


def get_memory_snapshot() -> MemorySnapshot:
    vm = _run("vm_stat")
    mem = _run("sysctl -n hw.memsize").strip()
    page_size = 16384
    vals: dict[str, int] = {}
    for line in vm.splitlines():
        m = re.search(r"page size of (\d+) bytes", line)
        if m:
            page_size = int(m.group(1))
        cleaned = line.replace(".", "")
        m = re.match(r"([^:]+):\s+([0-9]+)", cleaned)
        if m:
            vals[m.group(1).strip()] = int(m.group(2))
    total = int(mem or 0) / 1024**3
    free_pages = vals.get("Pages free", 0) + vals.get("Pages speculative", 0) + vals.get("Pages purgeable", 0)
    available = free_pages * page_size / 1024**3
    inactive = vals.get("Pages inactive", 0) * page_size / 1024**3
    return MemorySnapshot(total_gb=round(total, 2), available_gb=round(available, 2), available_plus_reclaimable_gb=round(available + inactive * 0.45, 2))
