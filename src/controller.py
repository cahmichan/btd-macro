"""High-level Roblox window, launcher, process, click, and anti-AFK controller."""

from __future__ import annotations

import os
import time
from typing import Optional

import psutil

from src.config import AppConfig
from src.win32_utils import (
    click_screen_point,
    drag_screen_points,
    enforce_window_layout,
    find_pids_by_name,
    find_roblox_hwnd,
    focus_window,
    get_client_rect_screen,
    is_process_running,
    move_mouse_abs,
    move_mouse_relative,
    scroll_mouse_wheel,
    send_key_tap,
)

ROBLOX_CLIENT_EXE = "RobloxPlayerBeta.exe"
ROBLOX_LAUNCHER_EXE = "RobloxPlayerLauncher.exe"


class RobloxController:
    """Encapsulates Roblox process lifecycle, window layout, and focus-verified inputs."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def is_roblox_running(self) -> bool:
        """Return True if RobloxPlayerBeta.exe is currently running."""
        return is_process_running(ROBLOX_CLIENT_EXE)

    def is_launcher_running(self) -> bool:
        """Return True if RobloxPlayerLauncher.exe (updater/installer) is running."""
        return is_process_running(ROBLOX_LAUNCHER_EXE)

    def get_hwnd(self) -> Optional[int]:
        """Return the active RobloxPlayerBeta.exe HWND, if visible."""
        return find_roblox_hwnd(ROBLOX_CLIENT_EXE)

    def get_client_rect(self, hwnd: Optional[int] = None) -> Optional[tuple[int, int, int, int]]:
        """Return (left, top, width, height) of the Roblox client area in screen pixels."""
        target = hwnd if hwnd is not None else self.get_hwnd()
        if target is None:
            return None
        return get_client_rect_screen(target)

    def ensure_window_layout(self, hwnd: Optional[int] = None) -> bool:
        """Snap the Roblox window to the configured fixed position and size."""
        target = hwnd if hwnd is not None else self.get_hwnd()
        if target is None:
            return False
        w = self.config.window
        return enforce_window_layout(
            target,
            x=w.x,
            y=w.y,
            width=w.width,
            height=w.height,
            size_mode=w.size_mode,
        )

    def focus(self, hwnd: Optional[int] = None) -> bool:
        """Bring the Roblox window to the foreground."""
        target = hwnd if hwnd is not None else self.get_hwnd()
        if target is None:
            return False
        return focus_window(target)

    def launch_game(self) -> bool:
        """Launch Build a Tower Defense directly via the Roblox deep-link URI."""
        uri = f"roblox://experiences/start?placeId={self.config.place_id}"
        try:
            os.startfile(uri)
            return True
        except OSError:
            return False

    def terminate_roblox(self, timeout_sec: float = 5.0) -> int:
        """Cleanly terminate all RobloxPlayerBeta.exe processes, killing stubborn survivors."""
        pids = find_pids_by_name(ROBLOX_CLIENT_EXE)
        if not pids:
            return 0

        procs: list[psutil.Process] = []
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                procs.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if not procs:
            return 0

        _, alive = psutil.wait_procs(procs, timeout=timeout_sec)
        for proc in alive:
            try:
                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if alive:
            psutil.wait_procs(alive, timeout=2.0)
        return len(procs)

    def click_client_point(
        self,
        client_x: int,
        client_y: int,
        hwnd: Optional[int] = None,
        dry_run: bool = False,
    ) -> bool:
        """Focus Roblox, convert client coords to screen coords, and click with hover nudge."""
        if dry_run:
            return True

        target = hwnd if hwnd is not None else self.get_hwnd()
        if target is None or not self.focus(target):
            return False

        rect = self.get_client_rect(target)
        if rect is None:
            return False

        left, top, width, height = rect
        if client_x < 0 or client_y < 0 or client_x >= width or client_y >= height:
            return False

        click_screen_point(left + int(client_x), top + int(client_y), hover_nudge_px=2)
        # Park cursor in a neutral area so hover effects don't linger on buttons
        move_mouse_abs(left + 20, top + max(20, int(height * 0.5)))
        return True

    def scroll_client_point(
        self,
        client_x: int,
        client_y: int,
        notches: int,
        hwnd: Optional[int] = None,
        dry_run: bool = False,
    ) -> bool:
        """Focus Roblox, move to client coordinates, and send mouse wheel notches."""
        if dry_run:
            return True

        target = hwnd if hwnd is not None else self.get_hwnd()
        if target is None or not self.focus(target):
            return False

        rect = self.get_client_rect(target)
        if rect is None:
            return False

        left, top, width, height = rect
        if client_x < 0 or client_y < 0 or client_x >= width or client_y >= height:
            return False

        move_mouse_abs(left + int(client_x), top + int(client_y))
        time.sleep(0.03)
        scroll_mouse_wheel(notches)
        return True

    def drag_client_points(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        hwnd: Optional[int] = None,
        dry_run: bool = False,
    ) -> bool:
        """Focus Roblox and drag cleanly between two clamped client-area coordinates."""
        if dry_run:
            return True

        target = hwnd if hwnd is not None else self.get_hwnd()
        if target is None or not self.focus(target):
            return False

        rect = self.get_client_rect(target)
        if rect is None:
            return False

        left, top, width, height = rect
        if width <= 8 or height <= 8:
            return False

        cx1 = max(4, min(width - 5, int(x1)))
        cy1 = max(4, min(height - 5, int(y1)))
        cx2 = max(4, min(width - 5, int(x2)))
        cy2 = max(4, min(height - 5, int(y2)))
        drag_screen_points(left + cx1, top + cy1, left + cx2, top + cy2)
        return True

    def park_cursor_neutral(self, hwnd: Optional[int] = None, dry_run: bool = False) -> bool:
        """Move cursor to top-left empty margin so hover tooltips never block hex tiles."""
        if dry_run:
            return True

        target = hwnd if hwnd is not None else self.get_hwnd()
        if target is None:
            return False

        rect = self.get_client_rect(target)
        if rect is None:
            return False

        left, top, width, height = rect
        if width <= 0 or height <= 0:
            return False

        park_x = left + min(20, max(0, width - 1))
        park_y = top + min(120, max(0, height - 1))
        move_mouse_abs(park_x, park_y)
        return True

    def send_key(self, key: str, hwnd: Optional[int] = None, dry_run: bool = False) -> bool:
        """Focus Roblox and send a single key tap."""
        if dry_run:
            return True

        target = hwnd if hwnd is not None else self.get_hwnd()
        if target is None or not self.focus(target):
            return False

        return send_key_tap(key)

    def send_anti_afk(self, hwnd: Optional[int] = None, dry_run: bool = False) -> bool:
        """Send a focus-verified neutral input to reset Roblox's 20-minute idle timer."""
        if dry_run or not self.config.anti_afk.enabled:
            return True

        target = hwnd if hwnd is not None else self.get_hwnd()
        if target is None or not self.focus(target):
            return False

        if self.config.anti_afk.method == "mouse_nudge":
            move_mouse_relative(3, 0)
            move_mouse_relative(-3, 0)
            return True

        return send_key_tap(self.config.anti_afk.key)
