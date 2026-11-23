"""Win32 utilities for DPI awareness, Roblox window control, SendInput, sleep lock, and F8 hotkey."""

from __future__ import annotations

import ctypes
import threading
import time
from ctypes import wintypes
from typing import Callable, Optional

import psutil
import win32con
import win32gui
import win32process

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# 64-bit safe pointer-sized integer for SendInput structs
ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32

# SendInput constants
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_ABSOLUTE = 0x8000

WHEEL_DELTA = 120

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

MAPVK_VK_TO_VSC = 0

# Sleep prevention constants
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

# Hotkey / message loop constants
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
ERROR_ALREADY_EXISTS = 183

# Virtual-key map for common keys and F1-F12 hotkeys
VK_MAP: dict[str, int] = {
    "space": 0x20,
    "tab": 0x09,
    "enter": 0x0D,
    "return": 0x0D,
    "esc": 0x1B,
    "escape": 0x1B,
    "shift": 0x10,
    "ctrl": 0x11,
    "alt": 0x12,
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
    "w": 0x57,
    "a": 0x41,
    "s": 0x53,
    "d": 0x44,
    "e": 0x45,
    "q": 0x51,
    "r": 0x52,
    **{f"f{i}": 0x6F + i for i in range(1, 13)},
}

EXTENDED_VKS = {0x25, 0x26, 0x27, 0x28}


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _anonymous_ = ("_input",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("_input", _INPUT_UNION),
    ]


user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT
user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
user32.MapVirtualKeyW.restype = wintypes.UINT
user32.IsZoomed.argtypes = [wintypes.HWND]
user32.IsZoomed.restype = wintypes.BOOL


def enable_dpi_awareness() -> None:
    """Enable Per-Monitor V2 DPI awareness before any GUI or screen coordinate calls."""
    # SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4)
    try:
        fn = user32.SetProcessDpiAwarenessContext
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = wintypes.BOOL
        if fn(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass

    # Fallback: Shcore SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE = 2)
    try:
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        shcore.SetProcessDpiAwareness.argtypes = [ctypes.c_int]
        shcore.SetProcessDpiAwareness.restype = ctypes.c_long
        shcore.SetProcessDpiAwareness(2)
        return
    except (AttributeError, OSError):
        pass

    # Legacy fallback
    try:
        user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def find_pids_by_name(process_name: str) -> list[int]:
    """Return all active PIDs matching process_name (case-insensitive)."""
    target = process_name.lower()
    pids: list[int] = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            name = proc.info.get("name") or ""
            if name.lower() == target:
                pids.append(int(proc.info["pid"]))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return pids


def is_process_running(process_name: str) -> bool:
    """Return True if at least one process with process_name is running."""
    return len(find_pids_by_name(process_name)) > 0


def find_roblox_hwnd(process_name: str = "RobloxPlayerBeta.exe") -> Optional[int]:
    """Find the primary visible HWND owned by RobloxPlayerBeta.exe (never browser tabs)."""
    pids = set(find_pids_by_name(process_name))
    if not pids:
        return None

    candidates: list[tuple[int, int, int]] = []  # (is_unowned, area, hwnd)

    def _enum_cb(hwnd: int, _: int) -> bool:
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if pid not in pids:
                return True
            rect = win32gui.GetClientRect(hwnd)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            if width <= 0 or height <= 0:
                return True
            owner = win32gui.GetWindow(hwnd, win32con.GW_OWNER)
            is_unowned = 1 if owner == 0 else 0
            candidates.append((is_unowned, width * height, hwnd))
        except Exception:
            return True
        return True

    try:
        win32gui.EnumWindows(_enum_cb, 0)
    except Exception:
        return None
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def get_client_rect_screen(hwnd: int) -> Optional[tuple[int, int, int, int]]:
    """Return (left, top, width, height) of hwnd's client area in physical screen pixels."""
    if not hwnd or not win32gui.IsWindow(hwnd) or win32gui.IsIconic(hwnd):
        return None
    try:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            return None
        screen_left, screen_top = win32gui.ClientToScreen(hwnd, (left, top))
        return (int(screen_left), int(screen_top), int(width), int(height))
    except Exception:
        return None


def get_primary_screen_size() -> tuple[int, int]:
    """Return (width, height) of the primary monitor in physical pixels."""
    return (
        int(user32.GetSystemMetrics(win32con.SM_CXSCREEN)),
        int(user32.GetSystemMetrics(win32con.SM_CYSCREEN)),
    )


def enforce_window_layout(
    hwnd: int,
    x: int = 0,
    y: int = 0,
    width: int = 1280,
    height: int = 720,
    size_mode: str = "window",
) -> bool:
    """Restore if minimized/maximized and enforce fixed position and dimensions."""
    if not hwnd or not win32gui.IsWindow(hwnd):
        return False
    try:
        if win32gui.IsIconic(hwnd) or bool(user32.IsZoomed(hwnd)):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            time.sleep(0.05)

        target_w = int(width)
        target_h = int(height)

        if size_mode == "client":
            style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
            ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            rect = wintypes.RECT(0, 0, target_w, target_h)
            if user32.AdjustWindowRectEx(ctypes.byref(rect), style, False, ex_style):
                target_w = rect.right - rect.left
                target_h = rect.bottom - rect.top

        win32gui.SetWindowPos(
            hwnd,
            0,
            int(x),
            int(y),
            target_w,
            target_h,
            win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE,
        )
        return True
    except Exception:
        return False


def is_window_focused(hwnd: int) -> bool:
    """Return True if hwnd is currently the active foreground window."""
    if not hwnd or not win32gui.IsWindow(hwnd):
        return False
    return int(win32gui.GetForegroundWindow()) == int(hwnd)


def focus_window(hwnd: int, retries: int = 3) -> bool:
    """Reliably bring hwnd to the foreground, using AttachThreadInput fallback if needed."""
    if not hwnd or not win32gui.IsWindow(hwnd):
        return False
    if is_window_focused(hwnd):
        return True

    for _ in range(max(1, retries)):
        try:
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            if is_window_focused(hwnd):
                return True
        except Exception:
            pass

        fg_hwnd = win32gui.GetForegroundWindow()
        cur_tid = kernel32.GetCurrentThreadId()
        fg_tid = win32process.GetWindowThreadProcessId(fg_hwnd)[0] if fg_hwnd else 0
        target_tid = win32process.GetWindowThreadProcessId(hwnd)[0]

        attached_fg = False
        attached_target = False
        try:
            if fg_tid and fg_tid != cur_tid:
                attached_fg = bool(user32.AttachThreadInput(cur_tid, fg_tid, True))
            if target_tid and target_tid != cur_tid:
                attached_target = bool(user32.AttachThreadInput(cur_tid, target_tid, True))
            win32gui.BringWindowToTop(hwnd)
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass
        finally:
            if attached_target:
                user32.AttachThreadInput(cur_tid, target_tid, False)
            if attached_fg:
                user32.AttachThreadInput(cur_tid, fg_tid, False)

        if is_window_focused(hwnd):
            return True
        time.sleep(0.05)

    return is_window_focused(hwnd)


def _send_inputs(*inputs: INPUT) -> int:
    arr = (INPUT * len(inputs))(*inputs)
    return int(user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT)))


def move_mouse_abs(screen_x: int, screen_y: int) -> None:
    """Move cursor to physical virtual-desktop screen coordinates via SendInput."""
    vx = int(user32.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN))
    vy = int(user32.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN))
    vw = max(2, int(user32.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)))
    vh = max(2, int(user32.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)))

    nx = int(round((screen_x - vx) * 65535 / (vw - 1)))
    ny = int(round((screen_y - vy) * 65535 / (vh - 1)))
    nx = max(0, min(65535, nx))
    ny = max(0, min(65535, ny))

    inp = INPUT(
        type=INPUT_MOUSE,
        mi=MOUSEINPUT(
            dx=nx,
            dy=ny,
            mouseData=0,
            dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
            time=0,
            dwExtraInfo=0,
        ),
    )
    _send_inputs(inp)


def move_mouse_relative(dx: int, dy: int) -> None:
    """Move cursor by a relative pixel delta via SendInput."""
    inp = INPUT(
        type=INPUT_MOUSE,
        mi=MOUSEINPUT(
            dx=int(dx),
            dy=int(dy),
            mouseData=0,
            dwFlags=MOUSEEVENTF_MOVE,
            time=0,
            dwExtraInfo=0,
        ),
    )
    _send_inputs(inp)


def click_screen_point(screen_x: int, screen_y: int, hover_nudge_px: int = 2) -> None:
    """Move to (screen_x, screen_y), perform a small hover nudge for Roblox UI, and click."""
    move_mouse_abs(screen_x, screen_y)
    time.sleep(0.04)

    if hover_nudge_px > 0:
        move_mouse_relative(hover_nudge_px, 0)
        time.sleep(0.03)
        move_mouse_relative(-hover_nudge_px, 0)
        time.sleep(0.04)

    down = INPUT(
        type=INPUT_MOUSE,
        mi=MOUSEINPUT(
            dx=0,
            dy=0,
            mouseData=0,
            dwFlags=MOUSEEVENTF_LEFTDOWN,
            time=0,
            dwExtraInfo=0,
        ),
    )
    up = INPUT(
        type=INPUT_MOUSE,
        mi=MOUSEINPUT(
            dx=0,
            dy=0,
            mouseData=0,
            dwFlags=MOUSEEVENTF_LEFTUP,
            time=0,
            dwExtraInfo=0,
        ),
    )
    _send_inputs(down)
    time.sleep(0.05)
    _send_inputs(up)


def scroll_mouse_wheel(notches: int, step_delay_sec: float = 0.03) -> None:
    """Scroll the vertical mouse wheel via SendInput (positive=up/zoom-in, negative=down/zoom-out)."""
    count = abs(int(notches))
    if count == 0:
        return

    direction = 1 if notches > 0 else -1
    delta_dword = (direction * WHEEL_DELTA) & 0xFFFFFFFF

    for idx in range(count):
        inp = INPUT(
            type=INPUT_MOUSE,
            mi=MOUSEINPUT(
                dx=0,
                dy=0,
                mouseData=delta_dword,
                dwFlags=MOUSEEVENTF_WHEEL,
                time=0,
                dwExtraInfo=0,
            ),
        )
        _send_inputs(inp)
        if step_delay_sec > 0 and idx + 1 < count:
            time.sleep(step_delay_sec)


def drag_screen_points(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    steps: int = 12,
    duration_sec: float = 0.25,
) -> None:
    """Smoothly drag with the left mouse button between two screen coordinates via SendInput."""
    start_x, start_y = int(x1), int(y1)
    end_x, end_y = int(x2), int(y2)
    move_mouse_abs(start_x, start_y)
    time.sleep(0.03)

    down = INPUT(
        type=INPUT_MOUSE,
        mi=MOUSEINPUT(
            dx=0,
            dy=0,
            mouseData=0,
            dwFlags=MOUSEEVENTF_LEFTDOWN,
            time=0,
            dwExtraInfo=0,
        ),
    )
    up = INPUT(
        type=INPUT_MOUSE,
        mi=MOUSEINPUT(
            dx=0,
            dy=0,
            mouseData=0,
            dwFlags=MOUSEEVENTF_LEFTUP,
            time=0,
            dwExtraInfo=0,
        ),
    )
    _send_inputs(down)
    time.sleep(0.03)

    num_steps = max(1, int(steps))
    step_sleep = max(0.0, float(duration_sec)) / num_steps
    for idx in range(1, num_steps + 1):
        frac = idx / num_steps
        ix = int(round(start_x + (end_x - start_x) * frac))
        iy = int(round(start_y + (end_y - start_y) * frac))
        move_mouse_abs(ix, iy)
        if step_sleep > 0:
            time.sleep(step_sleep)

    time.sleep(0.03)
    _send_inputs(up)


def resolve_virtual_key(key: str) -> Optional[int]:
    """Resolve a human-readable key name (e.g., 'space', 'F8', 'w') to a Win32 VK code."""
    normalized = key.strip().lower()
    if normalized in VK_MAP:
        return VK_MAP[normalized]
    if len(normalized) == 1 and normalized.isalnum():
        return ord(normalized.upper())
    return None


def send_key_tap(key: str, hold_sec: float = 0.06) -> bool:
    """Send a hardware-scancode key press and release via SendInput."""
    vk = resolve_virtual_key(key)
    if vk is None:
        return False

    scan = int(user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC))
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_EXTENDEDKEY if vk in EXTENDED_VKS else 0)

    down = INPUT(
        type=INPUT_KEYBOARD,
        ki=KEYBDINPUT(wVk=0, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=0),
    )
    up = INPUT(
        type=INPUT_KEYBOARD,
        ki=KEYBDINPUT(wVk=0, wScan=scan, dwFlags=flags | KEYEVENTF_KEYUP, time=0, dwExtraInfo=0),
    )
    sent_down = _send_inputs(down)
    time.sleep(max(0.01, hold_sec))
    sent_up = _send_inputs(up)
    return sent_down == 1 and sent_up == 1


def prevent_sleep() -> bool:
    """Keep Windows awake and display active during overnight macro runs."""
    res = kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
    return res != 0


def allow_sleep() -> bool:
    """Release the sleep-prevention execution state lock."""
    res = kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    return res != 0


class GlobalHotkeyListener:
    """Win32 RegisterHotKey listener on a background daemon thread (e.g., F8 kill switch)."""

    def __init__(self, hotkey: str, callback: Callable[[], None], hotkey_id: int = 1) -> None:
        self.hotkey = hotkey
        self.callback = callback
        self.hotkey_id = hotkey_id
        self._thread: Optional[threading.Thread] = None
        self._thread_id: Optional[int] = None
        self._ready = threading.Event()
        self._registered = False

    def start(self) -> bool:
        """Start the hotkey listener thread and wait briefly for registration."""
        if self._thread is not None and self._thread.is_alive():
            return self._registered
        vk = resolve_virtual_key(self.hotkey)
        if vk is None:
            return False

        self._ready.clear()
        self._thread = threading.Thread(
            target=self._message_loop,
            args=(vk,),
            name=f"Hotkey-{self.hotkey}",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=2.0)
        return self._registered

    def _message_loop(self, vk: int) -> None:
        self._thread_id = int(kernel32.GetCurrentThreadId())
        self._registered = bool(user32.RegisterHotKey(None, self.hotkey_id, 0, vk))
        self._ready.set()
        if not self._registered:
            return

        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY and msg.wParam == self.hotkey_id:
                    try:
                        self.callback()
                    except Exception:
                        pass
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        finally:
            user32.UnregisterHotKey(None, self.hotkey_id)
            self._registered = False

    def stop(self) -> None:
        """Unregister the hotkey and stop the background message loop."""
        if self._thread_id is not None:
            try:
                user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            except Exception:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        self._thread_id = None


class SingleInstanceGuard:
    """Named Win32 mutex preventing duplicate macro instances."""

    def __init__(self, mutex_name: str = "Local\\BtdOvernightMacroMutex") -> None:
        self.mutex_name = mutex_name
        self._handle: Optional[int] = None

    def acquire(self) -> bool:
        """Acquire the mutex; return False if another instance already holds it."""
        if self._handle is not None:
            return True
        handle = kernel32.CreateMutexW(None, False, self.mutex_name)
        if not handle:
            return True
        if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            return False
        self._handle = int(handle)
        return True

    def release(self) -> None:
        """Release and close the underlying Win32 mutex handle."""
        if self._handle is not None:
            kernel32.CloseHandle(self._handle)
            self._handle = None
