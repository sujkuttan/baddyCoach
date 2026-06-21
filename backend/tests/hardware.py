"""Hardware detection for test suite."""
import platform


def get_ram_gb() -> float:
    """Return total system RAM in GB."""
    if platform.system() == "Linux":
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / (1024**2)
    elif platform.system() == "Darwin":
        import subprocess
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        return int(result.strip()) / (1024**3)
    elif platform.system() == "Windows":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        c_ulonglong = ctypes.c_ulonglong

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", c_ulonglong),
                ("ullAvailPhys", c_ulonglong),
                ("ullTotalPageFile", c_ulonglong),
                ("ullAvailPageFile", c_ulonglong),
                ("ullTotalVirtual", c_ulonglong),
                ("ullAvailVirtual", c_ulonglong),
                ("ullAvailExtendedVirtual", c_ulonglong),
            ]

        meminfo = MEMORYSTATUSEX()
        meminfo.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        kernel32.GlobalMemoryStatusEx(ctypes.byref(meminfo))
        return meminfo.ullTotalPhys / (1024**3)
    return 8.0


def get_vram_gb() -> float | None:
    """Return GPU VRAM in GB, or None if no CUDA GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_mem / (1024**3)
    except Exception:
        pass
    return None


def has_gpu() -> bool:
    return get_vram_gb() is not None


def get_machine_tier() -> str:
    """Classify machine for test selection."""
    vram = get_vram_gb()
    ram = get_ram_gb()
    if vram and vram >= 12:
        return "high"
    if vram and vram >= 6:
        return "medium"
    if ram >= 16:
        return "medium"
    if ram >= 8:
        return "low"
    return "minimal"
