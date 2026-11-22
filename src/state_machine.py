"""Background-thread Finite State Machine (FSM) for the BTD overnight macro."""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from src.config import AppConfig
from src.controller import RobloxController
from src.vision import REQUIRED_TEMPLATES, VisionEngine
from src.win32_utils import allow_sleep, prevent_sleep


class MacroState(str, Enum):
    """Application states for the macro lifecycle."""

    IDLE = "IDLE"
    LAUNCHING = "LAUNCHING"
    READY_TO_START = "READY_TO_START"
    IN_RUN = "IN_RUN"
    DISCONNECTED = "DISCONNECTED"
    WATCHDOG_RECOVERY = "WATCHDOG_RECOVERY"


@dataclass
class MacroStats:
    """Live counters and confidence metrics reported to the UI."""

    runs_completed: int = 0
    disconnects: int = 0
    watchdog_recoveries: int = 0
    consecutive_recoveries: int = 0
    circuit_breaker_tripped: bool = False
    last_confidences: dict[str, float] = field(
        default_factory=lambda: {
            **{name: 0.0 for name in REQUIRED_TEMPLATES},
            "resume": 0.0,
            "hud_in_run": 0.0,
        }
    )


class MacroStateMachine:
    """Thread-safe FSM orchestrating launching, battle looping, anti-AFK, and recovery."""

    def __init__(
        self,
        config: AppConfig,
        vision: VisionEngine,
        controller: RobloxController,
        event_queue: Optional[queue.Queue[dict[str, Any]]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep_prevent_fn: Callable[[], bool] = prevent_sleep,
        sleep_allow_fn: Callable[[], bool] = allow_sleep,
    ) -> None:
        self.config = config
        self.vision = vision
        self.controller = controller
        self.event_queue = event_queue
        self.clock = clock
        self.sleep_prevent_fn = sleep_prevent_fn
        self.sleep_allow_fn = sleep_allow_fn

        self.stop_event = threading.Event()
        self.state = MacroState.IDLE
        self.dry_run = False
        self.stats = MacroStats()

        self.started_at: Optional[float] = None
        self.state_entered_at: float = self.clock()

        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Per-state tracking
        self._launch_initiated = False
        self._last_launch_attempt_at = 0.0
        self._update_grace_active = False
        self._click_attempts = 0
        self._last_anti_afk_at = self.clock()
        self._disconnect_handled = False
        self._disconnect_reason = "disconnected"
        self._reconnect_attempts = 0
        self._watchdog_reason = "watchdog_timeout"
        self._last_frame: Optional[np.ndarray] = None
        self._recovery_timestamps: list[float] = []
        self._hud_missing_streak = 0

    @property
    def is_running(self) -> bool:
        """Return True while the FSM background thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def get_uptime_sec(self) -> float:
        """Return elapsed session runtime in seconds, or 0.0 when stopped."""
        if self.started_at is None or self.state == MacroState.IDLE:
            return 0.0
        return max(0.0, self.clock() - self.started_at)

    def start(self, dry_run: bool = False) -> bool:
        """Start the FSM on a background daemon thread."""
        with self._lock:
            if self.is_running:
                return False

            self.vision.reload_templates()
            missing = self.vision.missing_required_templates()
            if not dry_run and "btn_battle" in missing:
                self._emit_log(
                    "ERROR",
                    "Missing required template assets/btn_battle.png. Run calibrate.bat first.",
                )
                return False
            if missing:
                self._emit_log(
                    "WARN",
                    f"Missing template(s): {', '.join(missing)}. Run calibrate.bat for full coverage.",
                )
            resume_name, _, _ = self._active_resume_spec()
            if not self.vision.has_template(resume_name):
                self._emit_log(
                    "INFO",
                    f"Optional mid-run fallback template assets/{resume_name}.png not found "
                    "(capture via calibrate.bat to auto-resume after mid-run disconnects).",
                )
            if not self.vision.has_template("hud_in_run"):
                self._emit_log(
                    "INFO",
                    "Optional active-run template assets/hud_in_run.png not found "
                    "(capture the red STOP button via calibrate.bat so the macro can detect an active wave).",
                )

            self.dry_run = dry_run
            self.stop_event.clear()
            self.stats = MacroStats()
            self._recovery_timestamps.clear()
            self.started_at = self.clock()
            self._transition_to(MacroState.LAUNCHING)

            self._thread = threading.Thread(
                target=self._run_loop,
                name="MacroFSM",
                daemon=True,
            )
            self._thread.start()
            mode_label = "DRY-RUN (no clicks)" if dry_run else "LIVE"
            self._emit_log("INFO", f"Macro started in {mode_label} mode.")
            return True

    def stop(self, reason: str = "Stopped by user") -> None:
        """Signal the FSM to stop immediately via stop_event."""
        self.stop_event.set()
        self._emit_log("INFO", f"Stop requested: {reason}")
        thread = self._thread
        if thread is not None and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=3.0)

    def _run_loop(self) -> None:
        stop_reason = "Stopped"
        try:
            if not self.dry_run:
                self.sleep_prevent_fn()
            while not self.stop_event.is_set():
                delay = self.step()
                if delay is None:
                    if self.stats.circuit_breaker_tripped:
                        stop_reason = "Circuit breaker tripped"
                    break
                if delay > 0:
                    self.stop_event.wait(delay)
        except Exception as exc:
            stop_reason = f"Unhandled FSM error: {exc}"
            self._emit_log("ERROR", stop_reason)
        finally:
            self.sleep_allow_fn()
            if self.state != MacroState.IDLE:
                self._transition_to(MacroState.IDLE)
            self._emit({"type": "stopped", "reason": stop_reason})

    def step(self) -> Optional[float]:
        """Execute a single FSM tick and return seconds to wait before next tick (or None)."""
        if self.stop_event.is_set() or self.state == MacroState.IDLE:
            return None

        if not self.dry_run and self.state != MacroState.WATCHDOG_RECOVERY:
            timeout_limit = self._current_state_timeout()
            if timeout_limit is not None:
                elapsed = self.clock() - self.state_entered_at
                if elapsed > timeout_limit:
                    self._watchdog_reason = f"{self.state.value.lower()}_timeout"
                    self._emit_log(
                        "WARN",
                        f"Watchdog timeout in {self.state.value} after {elapsed:.1f}s (limit {timeout_limit:.1f}s).",
                    )
                    self._transition_to(MacroState.WATCHDOG_RECOVERY)
                    return 0.0

        if self.state == MacroState.LAUNCHING:
            return self._step_launching()
        if self.state == MacroState.READY_TO_START:
            return self._step_ready_to_start()
        if self.state == MacroState.IN_RUN:
            return self._step_in_run()
        if self.state == MacroState.DISCONNECTED:
            return self._step_disconnected()
        if self.state == MacroState.WATCHDOG_RECOVERY:
            return self._step_watchdog_recovery()
        return None

    def _current_state_timeout(self) -> Optional[float]:
        t = self.config.timeouts
        if self.state == MacroState.LAUNCHING:
            return t.launching_sec + (t.launching_update_grace_sec if self._update_grace_active else 0.0)
        if self.state == MacroState.READY_TO_START:
            return t.ready_to_start_sec
        if self.state == MacroState.IN_RUN:
            return t.in_run_sec
        if self.state == MacroState.DISCONNECTED:
            return t.disconnected_sec
        return None

    def _transition_to(self, new_state: MacroState) -> None:
        self.state = new_state
        self.state_entered_at = self.clock()

        if new_state == MacroState.LAUNCHING:
            self._launch_initiated = False
            self._last_launch_attempt_at = 0.0
            self._update_grace_active = False
        elif new_state == MacroState.READY_TO_START:
            self._click_attempts = 0
        elif new_state == MacroState.IN_RUN:
            self._last_anti_afk_at = self.clock()
            self._hud_missing_streak = 0
        elif new_state == MacroState.DISCONNECTED:
            self._disconnect_handled = False
            self._reconnect_attempts = 0

        self._emit_state()

    def _capture_current_frame(self) -> tuple[Optional[int], Optional[np.ndarray]]:
        hwnd = self.controller.get_hwnd()
        if hwnd is None:
            return None, None
        rect = self.controller.get_client_rect(hwnd)
        if rect is None:
            return hwnd, None
        frame = self.vision.capture_client(rect)
        if frame is not None:
            self._last_frame = frame
        return hwnd, frame

    def _active_resume_spec(self) -> tuple[str, float, list[float]]:
        """Return (template_name, threshold, roi) for the configured mid-run disconnect mode."""
        th = self.config.thresholds
        rois = self.config.rois
        if self.config.recovery.resume_mode == "start_over":
            return ("btn_start_over", th.btn_start_over, rois.btn_start_over)
        return ("btn_yes_continue", th.btn_yes_continue, rois.btn_yes_continue)

    def _evaluate_core_templates(self, frame: np.ndarray) -> tuple[Any, Any, Any, Any, Any, Any]:
        th = self.config.thresholds
        rois = self.config.rois
        disc = self.vision.find(frame, "modal_disconnected", th.modal_disconnected, rois.modal_disconnected)
        battle = self.vision.find(frame, "btn_battle", th.btn_battle, rois.btn_battle)
        recon = self.vision.find(frame, "btn_reconnect", th.btn_reconnect, rois.btn_reconnect)
        resume_name, resume_th, resume_roi = self._active_resume_spec()
        resume_btn = self.vision.find(frame, resume_name, resume_th, resume_roi)
        cont_btn = self.vision.find(frame, "btn_continue", th.btn_continue, rois.btn_continue)
        in_run_hud = self.vision.find(frame, "hud_in_run", th.hud_in_run, rois.hud_in_run)
        self.stats.last_confidences = {
            "btn_battle": battle.confidence,
            "modal_disconnected": disc.confidence,
            "btn_reconnect": recon.confidence,
            "resume": resume_btn.confidence,
            "hud_in_run": in_run_hud.confidence,
        }
        self._emit_metrics()
        return disc, battle, recon, resume_btn, cont_btn, in_run_hud

    def _step_launching(self) -> Optional[float]:
        menu_poll = self.config.polling.menu_interval_sec
        hwnd = self.controller.get_hwnd()
        if hwnd is not None:
            rect = self.controller.get_client_rect(hwnd)
            if rect is not None:
                if not self.dry_run:
                    self.controller.ensure_window_layout(hwnd)
                    self.controller.focus(hwnd)
                    if self._launch_initiated and self.config.polling.post_launch_settle_sec > 0:
                        if self.stop_event.wait(self.config.polling.post_launch_settle_sec):
                            return None
                self._emit_log(
                    "INFO",
                    f"Roblox window ready ({self.config.window.width}x{self.config.window.height} at "
                    f"{self.config.window.x},{self.config.window.y}).",
                )
                self._transition_to(MacroState.READY_TO_START)
                return 0.0

        if self.dry_run:
            self._emit_log("WARN", "Dry-Run: waiting for an open RobloxPlayerBeta.exe window...")
            return menu_poll

        if self.controller.is_launcher_running():
            if not self._update_grace_active:
                self._update_grace_active = True
                self._emit_log(
                    "INFO",
                    "RobloxPlayerLauncher.exe detected (update in progress); extending launch grace timeout.",
                )
            return menu_poll

        if not self.controller.is_roblox_running():
            now = self.clock()
            if not self._launch_initiated or (now - self._last_launch_attempt_at) >= 30.0:
                self._launch_initiated = True
                self._last_launch_attempt_at = now
                if self.controller.launch_game():
                    self._emit_log("INFO", f"Launched Roblox placeId={self.config.place_id}.")
                else:
                    self._emit_log("WARN", "Failed to invoke Roblox URI launcher; will retry.")

        return menu_poll

    def _step_ready_to_start(self) -> Optional[float]:
        menu_poll = self.config.polling.menu_interval_sec
        hwnd, frame = self._capture_current_frame()
        if frame is None:
            if self.dry_run:
                self._emit_log("WARN", "Dry-Run: unable to capture Roblox client frame.")
                return menu_poll
            if not self.controller.is_roblox_running():
                self._disconnect_reason = "process_exited_ready"
                self._transition_to(MacroState.DISCONNECTED)
                return 0.0
            return menu_poll

        disc, battle, recon, resume_btn, cont_btn, in_run_hud = self._evaluate_core_templates(frame)

        def _dry_run_summary() -> str:
            return (
                f"Dry-Run: Battle={battle.confidence:.3f}, InRun={in_run_hud.confidence:.3f}, "
                f"Resume={resume_btn.confidence:.3f}, Disc={disc.confidence:.3f}, Recon={recon.confidence:.3f}"
            )

        if disc.found:
            if self.dry_run:
                self._emit_log("INFO", _dry_run_summary())
                return menu_poll
            self._disconnect_reason = "modal_disconnected_ready"
            self._transition_to(MacroState.DISCONNECTED)
            return 0.0

        if resume_btn.found and resume_btn.center is not None:
            if self.dry_run:
                self._emit_log("INFO", _dry_run_summary())
                return menu_poll
            resume_name, resume_th, resume_roi = self._active_resume_spec()
            return self._click_and_confirm_run_start(
                hwnd=hwnd,
                center=resume_btn.center,
                template_name=resume_name,
                threshold=resume_th,
                roi=resume_roi,
                started_log=f"Mid-run fallback ({resume_name}) confirmed; entering IN_RUN.",
                focus_fail_reason=f"{resume_name}_click_focus_failed",
                unconfirmed_reason=f"{resume_name}_click_unconfirmed",
            )

        if battle.found and battle.center is not None:
            if self.dry_run:
                self._emit_log("INFO", _dry_run_summary())
                return menu_poll
            return self._click_and_confirm_run_start(
                hwnd=hwnd,
                center=battle.center,
                template_name="btn_battle",
                threshold=self.config.thresholds.btn_battle,
                roi=self.config.rois.btn_battle,
                started_log="Battle started (Battle button disappeared).",
                focus_fail_reason="battle_click_focus_failed",
                unconfirmed_reason="battle_click_unconfirmed",
            )

        if cont_btn.found and cont_btn.center is not None:
            if self.dry_run:
                self._emit_log("INFO", f"{_dry_run_summary()}, Continue={cont_btn.confidence:.3f}")
                return menu_poll
            self._emit_log(
                "INFO",
                f"Detected saved battle Continue button ({cont_btn.confidence:.3f}); opening prompt.",
            )
            self.controller.click_client_point(
                cont_btn.center[0],
                cont_btn.center[1],
                hwnd=hwnd,
                dry_run=False,
            )
            return menu_poll

        if in_run_hud.found:
            if self.dry_run:
                self._emit_log("INFO", _dry_run_summary())
            self._emit_log(
                "INFO",
                f"Active wave HUD detected (hud_in_run={in_run_hud.confidence:.3f}); switching to IN_RUN.",
            )
            self._transition_to(MacroState.IN_RUN)
            return self.config.polling.in_run_interval_sec

        dismiss = self.vision.find_dismiss_target(frame, self.config.thresholds.dismiss)
        if dismiss is not None and dismiss.center is not None:
            if self.dry_run:
                self._emit_log(
                    "INFO",
                    f"Dry-Run: matched {dismiss.template_name} ({dismiss.confidence:.3f}).",
                )
            else:
                self._emit_log(
                    "INFO",
                    f"Clicking popup {dismiss.template_name} ({dismiss.confidence:.3f}).",
                )
                self.controller.click_client_point(
                    dismiss.center[0],
                    dismiss.center[1],
                    hwnd=hwnd,
                    dry_run=False,
                )
        elif self.dry_run:
            self._emit_log("INFO", _dry_run_summary())

        return menu_poll

    def _click_and_confirm_run_start(
        self,
        hwnd: Optional[int],
        center: tuple[int, int],
        template_name: str,
        threshold: float,
        roi: list[float],
        started_log: str,
        focus_fail_reason: str,
        unconfirmed_reason: str,
    ) -> Optional[float]:
        """Click a start/resume target and confirm it disappears before entering IN_RUN."""
        menu_poll = self.config.polling.menu_interval_sec
        clicked = self.controller.click_client_point(
            center[0],
            center[1],
            hwnd=hwnd,
            dry_run=False,
        )
        if not clicked:
            self._click_attempts += 1
            self._emit_log(
                "WARN",
                f"Failed to focus/click {template_name} (attempt {self._click_attempts}/"
                f"{self.config.recovery.max_click_retries}).",
            )
            if self._click_attempts >= self.config.recovery.max_click_retries:
                self._watchdog_reason = focus_fail_reason
                self._transition_to(MacroState.WATCHDOG_RECOVERY)
                return 0.0
            return menu_poll

        return self._confirm_action_started(
            template_name=template_name,
            threshold=threshold,
            roi=roi,
            started_log=started_log,
            unconfirmed_reason=unconfirmed_reason,
        )

    def _confirm_action_started(
        self,
        template_name: str,
        threshold: float,
        roi: list[float],
        started_log: str,
        unconfirmed_reason: str,
    ) -> Optional[float]:
        """Poll briefly to verify the clicked button disappears before entering IN_RUN."""
        polling = self.config.polling
        poll_sec = max(0.001, polling.click_confirm_poll_sec)
        max_polls = max(1, math.ceil(polling.click_confirm_timeout_sec / poll_sec))
        deadline = self.clock() + polling.click_confirm_timeout_sec
        polls = 0
        while True:
            if self.stop_event.wait(poll_sec):
                return None
            polls += 1

            _, confirm_frame = self._capture_current_frame()
            if confirm_frame is not None:
                disc, battle, _, resume_btn, _, in_run_hud = self._evaluate_core_templates(confirm_frame)
                if disc.found:
                    self._disconnect_reason = "modal_disconnected_post_click"
                    self._transition_to(MacroState.DISCONNECTED)
                    return 0.0
                if template_name == "btn_battle" and resume_btn.found:
                    self._click_attempts = 0
                    self._emit_log(
                        "INFO",
                        "Continue prompt appeared after clicking Battle; handling in READY_TO_START.",
                    )
                    self._transition_to(MacroState.READY_TO_START)
                    return 0.0
                target = battle if template_name == "btn_battle" else resume_btn
                hud_ready = not self.vision.has_template("hud_in_run") or in_run_hud.found
                if not target.found and hud_ready:
                    self._click_attempts = 0
                    self._emit_log("INFO", started_log)
                    self._transition_to(MacroState.IN_RUN)
                    return polling.in_run_interval_sec

            if polls >= max_polls or self.clock() >= deadline:
                break

        self._click_attempts += 1
        self._emit_log(
            "WARN",
            f"{template_name} still visible after click (attempt {self._click_attempts}/"
            f"{self.config.recovery.max_click_retries}).",
        )
        if self._click_attempts >= self.config.recovery.max_click_retries:
            self._watchdog_reason = unconfirmed_reason
            self._transition_to(MacroState.WATCHDOG_RECOVERY)
            return 0.0
        return polling.menu_interval_sec

    def _step_in_run(self) -> Optional[float]:
        run_poll = self.config.polling.in_run_interval_sec

        if not self.controller.is_roblox_running():
            if self.dry_run:
                return run_poll
            self._last_frame = None
            self._disconnect_reason = "process_crashed_in_run"
            self._transition_to(MacroState.DISCONNECTED)
            return 0.0

        hwnd, frame = self._capture_current_frame()
        if frame is None:
            return run_poll

        disc, battle, recon, resume_btn, cont_btn, in_run_hud = self._evaluate_core_templates(frame)

        if self.dry_run:
            self._emit_log(
                "INFO",
                f"Dry-Run (IN_RUN): Battle={battle.confidence:.3f}, InRun={in_run_hud.confidence:.3f}, "
                f"Resume={resume_btn.confidence:.3f}, Disc={disc.confidence:.3f}, Recon={recon.confidence:.3f}",
            )
            if not in_run_hud.found and (battle.found or resume_btn.found or cont_btn.found):
                self._emit_log("INFO", "Dry-Run: wave HUD gone and menu visible; switching to READY_TO_START.")
                self._transition_to(MacroState.READY_TO_START)
                return self.config.polling.menu_interval_sec
            return run_poll

        if disc.found:
            self._disconnect_reason = "modal_disconnected_in_run"
            self._transition_to(MacroState.DISCONNECTED)
            return 0.0

        if resume_btn.found or cont_btn.found:
            self._emit_log(
                "WARN",
                "Ongoing-run resume UI detected mid-run; handling in READY_TO_START.",
            )
            self._transition_to(MacroState.READY_TO_START)
            return 0.0

        if battle.found:
            self._hud_missing_streak = 0
            self.stats.runs_completed += 1
            self.stats.consecutive_recoveries = 0
            self._recovery_timestamps.clear()
            self._emit_log(
                "INFO",
                f"Run #{self.stats.runs_completed} completed (Battle button reappeared).",
            )
            self._emit_metrics()
            self._transition_to(MacroState.READY_TO_START)
            return 0.0

        if self.vision.has_template("hud_in_run"):
            if in_run_hud.found:
                self._hud_missing_streak = 0
            else:
                self._hud_missing_streak += 1
                if self._hud_missing_streak >= 3:
                    self._emit_log(
                        "WARN",
                        f"Wave HUD missing for {self._hud_missing_streak} checks "
                        f"(Battle={battle.confidence:.3f}, InRun={in_run_hud.confidence:.3f}); "
                        "returning to READY_TO_START.",
                    )
                    self._transition_to(MacroState.READY_TO_START)
                    return 0.0

        if (
            self.config.anti_afk.enabled
            and (self.clock() - self._last_anti_afk_at) >= self.config.anti_afk.interval_sec
        ):
            if self.controller.send_anti_afk(hwnd=hwnd, dry_run=self.dry_run):
                self._last_anti_afk_at = self.clock()
                self._emit_log("INFO", "Sent anti-AFK input.")
            else:
                self._emit_log("WARN", "Failed to send anti-AFK input (window unavailable).")

        return run_poll

    def _step_disconnected(self) -> Optional[float]:
        menu_poll = self.config.polling.menu_interval_sec
        hwnd, frame = self._capture_current_frame()

        if not self._disconnect_handled:
            self._disconnect_handled = True
            self.stats.disconnects += 1
            snap = self.vision.save_failure_snapshot(
                frame if frame is not None else self._last_frame,
                self._disconnect_reason,
            )
            snap_msg = f" Saved snapshot {snap.name}." if snap is not None else ""
            self._emit_log("WARN", f"Disconnect handling started ({self._disconnect_reason}).{snap_msg}")
            self._emit_metrics()

        if not self.controller.is_roblox_running() or hwnd is None or frame is None:
            self._emit_log("WARN", "Roblox process/window unavailable; terminating for clean relaunch.")
            self.controller.terminate_roblox()
            self._transition_to(MacroState.LAUNCHING)
            return 0.0

        disc, battle, recon, resume_btn, cont_btn, in_run_hud = self._evaluate_core_templates(frame)

        if not disc.found and in_run_hud.found:
            self._emit_log("INFO", "Recovered from disconnect directly into active run (hud_in_run).")
            self._transition_to(MacroState.IN_RUN)
            return 0.0

        if not disc.found and (battle.found or resume_btn.found or cont_btn.found):
            self._emit_log("INFO", "Recovered from disconnect; battle/resume UI visible.")
            self._transition_to(MacroState.READY_TO_START)
            return 0.0

        if not disc.found and not recon.found:
            if self._reconnect_attempts > 0:
                return menu_poll
            self._emit_log("INFO", "Disconnect modal cleared; returning to READY_TO_START.")
            self._transition_to(MacroState.READY_TO_START)
            return 0.0

        if recon.found and recon.center is not None:
            max_retries = self.config.recovery.max_reconnect_retries
            if self._reconnect_attempts < max_retries:
                self._reconnect_attempts += 1
                self._emit_log(
                    "INFO",
                    f"Clicking Reconnect (attempt {self._reconnect_attempts}/{max_retries}).",
                )
                self.controller.click_client_point(
                    recon.center[0],
                    recon.center[1],
                    hwnd=hwnd,
                    dry_run=self.dry_run,
                )
                return self.config.recovery.reconnect_wait_sec

            self._emit_log(
                "WARN",
                "Reconnect retries exhausted; terminating Roblox for clean relaunch.",
            )
            self.controller.terminate_roblox()
            self._transition_to(MacroState.LAUNCHING)
            return 0.0

        self._emit_log(
            "WARN",
            "Leave-only disconnect modal detected (no Reconnect button); terminating Roblox for relaunch.",
        )
        self.controller.terminate_roblox()
        self._transition_to(MacroState.LAUNCHING)
        return 0.0

    def _step_watchdog_recovery(self) -> Optional[float]:
        now = self.clock()
        self.stats.watchdog_recoveries += 1

        _, frame = self._capture_current_frame()
        snap = self.vision.save_failure_snapshot(
            frame if frame is not None else self._last_frame,
            self._watchdog_reason,
        )
        snap_msg = f" Saved snapshot {snap.name}." if snap is not None else ""

        window_sec = self.config.recovery.circuit_breaker_window_sec
        self._recovery_timestamps = [
            ts for ts in self._recovery_timestamps if (now - ts) <= window_sec
        ]
        self._recovery_timestamps.append(now)
        self.stats.consecutive_recoveries = len(self._recovery_timestamps)
        self._emit_metrics()

        max_triggers = self.config.recovery.circuit_breaker_max_triggers
        if len(self._recovery_timestamps) >= max_triggers:
            self.stats.circuit_breaker_tripped = True
            self._emit_log(
                "ERROR",
                f"Circuit breaker tripped: {len(self._recovery_timestamps)} consecutive recoveries "
                f"within {window_sec:.0f}s ({self._watchdog_reason}).{snap_msg} Stopping cleanly.",
            )
            self.controller.terminate_roblox()
            self._transition_to(MacroState.IDLE)
            self.stop_event.set()
            return None

        self._emit_log(
            "WARN",
            f"Watchdog recovery #{self.stats.watchdog_recoveries} ({self._watchdog_reason}, "
            f"{self.stats.consecutive_recoveries}/{max_triggers} in window).{snap_msg} Relaunching Roblox.",
        )
        self.controller.terminate_roblox()
        self._transition_to(MacroState.LAUNCHING)
        return 0.0

    def _emit(self, event: dict[str, Any]) -> None:
        if self.event_queue is not None:
            try:
                self.event_queue.put_nowait(event)
            except queue.Full:
                pass

    def _emit_log(self, level: str, message: str) -> None:
        self._emit(
            {
                "type": "log",
                "level": level,
                "message": message,
                "ts": datetime.now().strftime("%H:%M:%S"),
            }
        )

    def _emit_state(self) -> None:
        self._emit(
            {
                "type": "state",
                "state": self.state.value,
                "dry_run": self.dry_run,
                "stats": self.stats,
            }
        )

    def _emit_metrics(self) -> None:
        self._emit(
            {
                "type": "metrics",
                "stats": self.stats,
                "confidences": dict(self.stats.last_confidences),
            }
        )
