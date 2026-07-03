from __future__ import annotations

import platform as _platform
import sys

_OS_NAMES = {"darwin": "darwin", "win32": "windows", "cygwin": "windows"}
_ARCH_NAMES = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}


def current_platform() -> str:
    """Return the machine selector, e.g. 'darwin-arm64' or 'windows-x86_64'."""
    if sys.platform in _OS_NAMES:
        os_name = _OS_NAMES[sys.platform]
    elif sys.platform.startswith("linux"):
        os_name = "linux"
    else:
        os_name = sys.platform
    arch = _platform.machine().lower()
    return f"{os_name}-{_ARCH_NAMES.get(arch, arch)}"


def platform_matches(entry: str, machine: str) -> bool:
    """Match a provider platform entry against a machine selector.

    Entries may be 'any', a bare OS ('darwin', 'windows', 'linux'), or a full
    os-arch pair ('darwin-arm64').
    """
    entry = str(entry).strip().lower()
    if entry in {"any", "*", "all"}:
        return True
    if entry == machine:
        return True
    return "-" not in entry and machine.split("-")[0] == entry
