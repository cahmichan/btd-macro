"""Compact CustomTkinter control panel docked beside the Roblox window."""

from __future__ import annotations

import os
import queue
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TextIO

import customtkinter as ctk

from src.config import AppConfig, get_logs_dir, save_config
from src.controller import RobloxController
from src.state_machine import MacroState, MacroStateMachine, MacroStats
from src.win32_utils import GlobalHotkeyListener, allow_sleep, get_primary_screen_size

STATE_COLORS: dict[str, str] = {
    MacroState.IDLE.value: "#6B7280",
    MacroState.LAUNCHING.value: "#2563EB",
    MacroState.READY_TO_START.value: "#D97706",
    MacroState.UPGRADING.value: "#0891B2",
    MacroState.IN_RUN.value: "#16A34A",
    MacroState.DISCONNECTED.value: "#DC2626",
    MacroState.WATCHDOG_RECOVERY.value: "#9333EA",
}


def format_duration(seconds: float) -> str:
    """Format elapsed seconds as HH:MM:SS."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class MacroApp(ctk.CTk):
    """Main desktop control panel communicating with the FSM via queue.Queue + after()."""

    MAX_UI_LOG_LINES = 500

    def __init__(
        self,
        config: AppConfig,
        fsm: MacroStateMachine,
        controller: RobloxController,
        event_queue: queue.Queue[dict[str, Any]],
        hotkey_listener: Optional[GlobalHotkeyListener] = None,
    ) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.config_data = config
        self.fsm = fsm
        self.controller = controller
        self.event_queue = event_queue
        self.hotkey_listener = hotkey_listener

        self._log_file: Optional[TextIO] = None
        self._log_path: Optional[Path] = None
        self._ui_line_count = 0
        self._closing = False

        self._init_session_log()
        self._configure_window()
        self._build_widgets()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.append_log(
            "INFO",
            f"Ready. Place ID {self.config_data.place_id} | Hotkey {self.config_data.hotkey} | "
            f"Log: {self._log_path.name if self._log_path else 'disabled'}",
        )
        missing = self.fsm.vision.missing_required_templates()
        if missing:
            self.append_log(
                "WARN",
                f"Missing templates: {', '.join(missing)}. Run calibrate.bat before starting.",
            )
        if self.config_data.auto_upgrade.enabled:
            self._warn_if_upgrade_templates_missing()

        self.after(100, self._poll_queue)

    def _init_session_log(self) -> None:
        logs_dir = get_logs_dir()
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = logs_dir / f"session_{stamp}.log"
        self._log_file = self._log_path.open("a", encoding="utf-8")

    def _configure_window(self) -> None:
        self.title("BTD Overnight Macro")
        panel_w = self.config_data.ui.panel_width
        panel_h = self.config_data.ui.panel_height
        gap = self.config_data.ui.panel_gap

        screen_w, screen_h = get_primary_screen_size()
        desired_x = self.config_data.window.x + self.config_data.window.width + gap
        desired_y = self.config_data.window.y

        # Clamp panel within screen bounds while avoiding overlap with Roblox when possible
        pos_x = max(0, min(desired_x, max(0, screen_w - panel_w)))
        pos_y = max(0, min(desired_y, max(0, screen_h - panel_h)))

        self.geometry(f"{panel_w}x{panel_h}+{pos_x}+{pos_y}")
        self.minsize(340, 560)

    def _build_widgets(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # Header
        header = ctk.CTkFrame(self)
        header.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="BTD Overnight Macro",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, padx=12, pady=(10, 2), sticky="w")

        ctk.CTkLabel(
            header,
            text=f"Place ID: {self.config_data.place_id}   |   Kill Switch: {self.config_data.hotkey}",
            font=ctk.CTkFont(size=12),
            text_color="#9CA3AF",
        ).grid(row=1, column=0, padx=12, pady=(0, 10), sticky="w")

        # Status & Metrics Card
        stats_frame = ctk.CTkFrame(self)
        stats_frame.grid(row=1, column=0, padx=12, pady=6, sticky="ew")
        stats_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(stats_frame, text="State", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=0, column=0, padx=12, pady=(10, 4), sticky="w"
        )
        self.state_badge = ctk.CTkLabel(
            stats_frame,
            text=MacroState.IDLE.value,
            fg_color=STATE_COLORS[MacroState.IDLE.value],
            corner_radius=6,
            padx=10,
            pady=3,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.state_badge.grid(row=0, column=1, padx=12, pady=(10, 4), sticky="e")

        self.runs_label = self._add_stat_row(stats_frame, 1, "Runs Completed", "0")
        self.upgrades_label = self._add_stat_row(stats_frame, 2, "Upgrades Bought", "0")
        self.disconnects_label = self._add_stat_row(stats_frame, 3, "Disconnects", "0")
        self.recoveries_label = self._add_stat_row(stats_frame, 4, "Watchdog Recoveries", "0")
        self.uptime_label = self._add_stat_row(stats_frame, 5, "Uptime", "00:00:00")

        self.conf_label = ctk.CTkLabel(
            stats_frame,
            text="Conf: Battle 0.00 | InRun 0.00 | Resume 0.00 | Disc 0.00",
            font=ctk.CTkFont(size=11),
            text_color="#9CA3AF",
        )
        self.conf_label.grid(row=6, column=0, columnspan=2, padx=12, pady=(4, 10), sticky="w")

        # Controls
        controls = ctk.CTkFrame(self)
        controls.grid(row=2, column=0, padx=12, pady=6, sticky="ew")
        controls.grid_columnconfigure((0, 1), weight=1)

        self.start_stop_btn = ctk.CTkButton(
            controls,
            text="Start Macro",
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#16A34A",
            hover_color="#15803D",
            command=self._on_toggle_start_stop,
        )
        self.start_stop_btn.grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 8), sticky="ew")

        self.dry_run_var = ctk.BooleanVar(value=False)
        self.dry_run_switch = ctk.CTkSwitch(
            controls,
            text="Dry-Run (Vision Test)",
            variable=self.dry_run_var,
        )
        self.dry_run_switch.grid(row=1, column=0, columnspan=2, padx=12, pady=(0, 8), sticky="w")

        self.auto_upgrade_var = ctk.BooleanVar(value=bool(self.config_data.auto_upgrade.enabled))
        self.auto_upgrade_switch = ctk.CTkSwitch(
            controls,
            text="Auto-Upgrade",
            variable=self.auto_upgrade_var,
            command=self._on_auto_upgrade_toggled,
        )
        self.auto_upgrade_switch.grid(row=2, column=0, columnspan=2, padx=12, pady=(0, 8), sticky="w")

        ctk.CTkLabel(
            controls,
            text="Upgrade Scope",
            font=ctk.CTkFont(size=11),
            text_color="#D1D5DB",
        ).grid(row=3, column=0, padx=12, pady=(0, 8), sticky="w")

        initial_scope = (
            "Main → Sides" if self.config_data.auto_upgrade.scope == "main_then_sides" else "Main Only"
        )
        self.upgrade_scope_seg = ctk.CTkSegmentedButton(
            controls,
            values=["Main Only", "Main → Sides"],
            command=self._on_upgrade_scope_changed,
        )
        self.upgrade_scope_seg.set(initial_scope)
        self.upgrade_scope_seg.grid(row=3, column=1, padx=(0, 12), pady=(0, 8), sticky="ew")

        ctk.CTkLabel(
            controls,
            text="Mid-Run Disconnect",
            font=ctk.CTkFont(size=11),
            text_color="#D1D5DB",
        ).grid(row=4, column=0, padx=12, pady=(0, 8), sticky="w")

        initial_mode = (
            "Start Over" if self.config_data.recovery.resume_mode == "start_over" else "Resume Run"
        )
        self.resume_mode_seg = ctk.CTkSegmentedButton(
            controls,
            values=["Resume Run", "Start Over"],
            command=self._on_resume_mode_changed,
        )
        self.resume_mode_seg.set(initial_mode)
        self.resume_mode_seg.grid(row=4, column=1, padx=(0, 12), pady=(0, 8), sticky="ew")

        self.snap_btn = ctk.CTkButton(
            controls,
            text="Snap Roblox 1280x720",
            height=30,
            fg_color="#374151",
            hover_color="#4B5563",
            command=self._on_snap_roblox,
        )
        self.snap_btn.grid(row=5, column=0, padx=(12, 6), pady=(0, 12), sticky="ew")

        self.logs_btn = ctk.CTkButton(
            controls,
            text="Open Logs",
            height=30,
            fg_color="#374151",
            hover_color="#4B5563",
            command=self._on_open_logs,
        )
        self.logs_btn.grid(row=5, column=1, padx=(6, 12), pady=(0, 12), sticky="ew")

        # Activity Log
        log_frame = ctk.CTkFrame(self)
        log_frame.grid(row=3, column=0, padx=12, pady=(6, 12), sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            log_frame,
            text="Activity Log",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=0, column=0, padx=12, pady=(8, 4), sticky="w")

        self.log_box = ctk.CTkTextbox(
            log_frame,
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="word",
            state="disabled",
        )
        self.log_box.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")

    @staticmethod
    def _add_stat_row(parent: ctk.CTkFrame, row: int, label: str, initial: str) -> ctk.CTkLabel:
        ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=12), text_color="#D1D5DB").grid(
            row=row, column=0, padx=12, pady=2, sticky="w"
        )
        val_label = ctk.CTkLabel(parent, text=initial, font=ctk.CTkFont(size=12, weight="bold"))
        val_label.grid(row=row, column=1, padx=12, pady=2, sticky="e")
        return val_label

    def _warn_if_upgrade_templates_missing(self) -> None:
        missing = [
            name
            for name in ("btn_upgrades", "btn_upgrades_close")
            if not self.fsm.vision.has_template(name)
        ]
        if missing:
            self.append_log(
                "WARN",
                f"Auto-Upgrade enabled, but missing template(s): {', '.join(missing)}. Run calibrate.bat to capture them.",
            )

    def _on_auto_upgrade_toggled(self) -> None:
        enabled = bool(self.auto_upgrade_var.get())
        self.config_data.auto_upgrade.enabled = enabled
        try:
            save_config(self.config_data)
        except OSError as exc:
            self.append_log("WARN", f"Could not save auto_upgrade.enabled to config.json: {exc}")
        self.append_log("INFO", f"Auto-Upgrade {'enabled' if enabled else 'disabled'}.")
        if enabled:
            self.fsm.vision.reload_templates()
            self._warn_if_upgrade_templates_missing()

    def _on_upgrade_scope_changed(self, value: str) -> None:
        scope = "main_then_sides" if value == "Main → Sides" else "main_only"
        self.config_data.auto_upgrade.scope = scope
        try:
            save_config(self.config_data)
        except OSError as exc:
            self.append_log("WARN", f"Could not save auto_upgrade.scope to config.json: {exc}")
        self.append_log("INFO", f"Auto-Upgrade scope set to '{scope}'.")

    def _on_resume_mode_changed(self, value: str) -> None:
        mode = "start_over" if value == "Start Over" else "continue"
        self.config_data.recovery.resume_mode = mode
        try:
            save_config(self.config_data)
        except OSError as exc:
            self.append_log("WARN", f"Could not save resume_mode to config.json: {exc}")
        self.append_log("INFO", f"Mid-run disconnect mode set to '{mode}'.")

    def _on_toggle_start_stop(self) -> None:
        if self.fsm.is_running:
            self.fsm.stop("Stop button clicked")
            self._set_running_ui(False)
        else:
            dry_run = bool(self.dry_run_var.get())
            if self.fsm.start(dry_run=dry_run):
                self._set_running_ui(True)

    def _on_snap_roblox(self) -> None:
        hwnd = self.controller.get_hwnd()
        if hwnd is None:
            self.append_log("WARN", "RobloxPlayerBeta.exe window not found to snap.")
            return
        if self.controller.ensure_window_layout(hwnd):
            rect = self.controller.get_client_rect(hwnd)
            self.append_log(
                "INFO",
                f"Snapped Roblox window to {self.config_data.window.width}x{self.config_data.window.height}. "
                f"Client rect: {rect}",
            )
        else:
            self.append_log("WARN", "Failed to resize/position Roblox window.")

    def _on_open_logs(self) -> None:
        logs_dir = get_logs_dir()
        logs_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(logs_dir))
        except OSError as exc:
            self.append_log("ERROR", f"Could not open logs folder: {exc}")

    def _set_running_ui(self, running: bool) -> None:
        if running:
            self.start_stop_btn.configure(
                text=f"Stop Macro ({self.config_data.hotkey})",
                fg_color="#DC2626",
                hover_color="#B91C1C",
            )
            self.dry_run_switch.configure(state="disabled")
            self.resume_mode_seg.configure(state="disabled")
        else:
            self.start_stop_btn.configure(
                text="Start Macro",
                fg_color="#16A34A",
                hover_color="#15803D",
            )
            self.dry_run_switch.configure(state="normal")
            self.resume_mode_seg.configure(state="normal")

    def _update_stats_ui(self, state_name: str, stats: MacroStats) -> None:
        color = STATE_COLORS.get(state_name, "#6B7280")
        badge_text = f"{state_name} (DRY)" if self.fsm.dry_run and state_name != MacroState.IDLE.value else state_name
        self.state_badge.configure(text=badge_text, fg_color=color)
        self.runs_label.configure(text=str(stats.runs_completed))
        self.upgrades_label.configure(text=str(stats.upgrades_purchased))
        self.disconnects_label.configure(text=str(stats.disconnects))
        self.recoveries_label.configure(text=str(stats.watchdog_recoveries))

        conf = stats.last_confidences
        self.conf_label.configure(
            text=(
                f"Conf: Battle {conf.get('btn_battle', 0.0):.2f} | "
                f"InRun {conf.get('hud_in_run', 0.0):.2f} | "
                f"Resume {conf.get('resume', 0.0):.2f} | "
                f"Disc {conf.get('modal_disconnected', 0.0):.2f}"
            )
        )

    def append_log(self, level: str, message: str, ts: Optional[str] = None) -> None:
        """Append a line to both the UI log box and the session log file."""
        stamp = ts or datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] [{level}] {message}\n"

        if self._log_file is not None and not self._log_file.closed:
            try:
                self._log_file.write(line)
                self._log_file.flush()
            except OSError:
                pass

        self.log_box.configure(state="normal")
        self.log_box.insert("end", line)
        self._ui_line_count += 1
        if self._ui_line_count > self.MAX_UI_LOG_LINES:
            excess = self._ui_line_count - self.MAX_UI_LOG_LINES
            self.log_box.delete("1.0", f"{excess + 1}.0")
            self._ui_line_count = self.MAX_UI_LOG_LINES
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _poll_queue(self) -> None:
        if self._closing:
            return

        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break

            etype = event.get("type")
            if etype == "log":
                self.append_log(
                    str(event.get("level", "INFO")),
                    str(event.get("message", "")),
                    ts=event.get("ts"),
                )
            elif etype == "state":
                self._update_stats_ui(str(event.get("state", "IDLE")), event["stats"])
            elif etype == "metrics":
                self._update_stats_ui(self.fsm.state.value, event["stats"])
            elif etype == "stopped":
                self._set_running_ui(False)
                self._update_stats_ui(MacroState.IDLE.value, self.fsm.stats)

        if self.fsm.is_running:
            self.uptime_label.configure(text=format_duration(self.fsm.get_uptime_sec()))

        self.after(100, self._poll_queue)

    def _on_close(self) -> None:
        self._closing = True
        try:
            if self.fsm.is_running:
                self.fsm.stop("Application closing")
        finally:
            if self.hotkey_listener is not None:
                self.hotkey_listener.stop()
            allow_sleep()
            if self._log_file is not None and not self._log_file.closed:
                self._log_file.close()
            self.destroy()
