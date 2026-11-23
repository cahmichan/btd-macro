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
from src.ocr import WindowsOcrEngine
from src.spawner_vision import (
    SpawnerBuyButton,
    SpawnerScanResult,
    detect_spawner_buy_buttons,
    is_scroll_grid_unchanged,
)
from src.upgrade_vision import (
    TreeScanResult,
    UpgradeCandidate,
    build_hud_mask,
    classify_upgrade_tooltip,
    detect_affordable_upgrades,
    extract_hover_tooltip_crop,
    find_empty_drag_origin,
    is_same_candidate,
)
from src.vision import REQUIRED_TEMPLATES, VisionEngine
from src.win32_utils import allow_sleep, prevent_sleep


class MacroState(str, Enum):
    """Application states for the macro lifecycle."""

    IDLE = "IDLE"
    LAUNCHING = "LAUNCHING"
    READY_TO_START = "READY_TO_START"
    PURCHASING_SPAWNERS = "PURCHASING_SPAWNERS"
    UPGRADING = "UPGRADING"
    IN_RUN = "IN_RUN"
    DISCONNECTED = "DISCONNECTED"
    WATCHDOG_RECOVERY = "WATCHDOG_RECOVERY"


@dataclass
class MacroStats:
    """Live counters and confidence metrics reported to the UI."""

    runs_completed: int = 0
    spawners_purchased: int = 0
    upgrades_purchased: int = 0
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
        upgrade_scanner: Callable[..., TreeScanResult] = detect_affordable_upgrades,
        spawner_scanner: Callable[[Optional[np.ndarray]], SpawnerScanResult] = detect_spawner_buy_buttons,
        ocr_engine: Optional[Any] = None,
        tooltip_extractor: Callable[..., Optional[np.ndarray]] = extract_hover_tooltip_crop,
    ) -> None:
        self.config = config
        self.vision = vision
        self.controller = controller
        self.event_queue = event_queue
        self.clock = clock
        self.sleep_prevent_fn = sleep_prevent_fn
        self.sleep_allow_fn = sleep_allow_fn
        self.upgrade_scanner = upgrade_scanner
        self.spawner_scanner = spawner_scanner
        self._ocr_engine = ocr_engine
        self._owns_ocr_engine = False
        self.tooltip_extractor = tooltip_extractor

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
        # Set when IN_RUN fell back to READY_TO_START because the wave HUD vanished (e.g. end-of-run
        # results screen). If Battle then appears in READY_TO_START, the run is counted as completed.
        self._run_end_unconfirmed = False
        self._last_refocus_at = float("-inf")
        self._pending_spawner_pass = False
        self._pending_upgrade_pass = False
        self._stop_reason: Optional[str] = None

    def _get_ocr_lines(self, crop: Optional[np.ndarray], *, invert: bool = True, scale: float = 3.0) -> list[str]:
        """Run OCR via injected callable/engine or lazily create WindowsOcrEngine."""
        if crop is None or crop.size == 0:
            return []
        engine = self._ocr_engine
        if engine is None:
            engine = WindowsOcrEngine()
            self._ocr_engine = engine
            self._owns_ocr_engine = True
        if callable(engine) and not hasattr(engine, "read_lines"):
            return list(engine(crop, invert=invert, scale=scale))
        return list(engine.read_lines(crop, invert=invert, scale=scale))

    def close(self) -> None:
        """Cleanly shut down any owned background OCR worker."""
        engine = self._ocr_engine
        if self._owns_ocr_engine and engine is not None and hasattr(engine, "close"):
            try:
                engine.close()
            except Exception:
                pass
            self._ocr_engine = None
            self._owns_ocr_engine = False

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
            if self.config.auto_spawner.enabled:
                missing_sp = [
                    t for t in ("btn_spawners", "btn_spawners_close") if not self.vision.has_template(t)
                ]
                if missing_sp:
                    self._emit_log(
                        "WARN",
                        f"Auto-Spawners is enabled, but missing template(s): {', '.join(missing_sp)}. "
                        "Run calibrate.bat to capture them.",
                    )
            if self.config.auto_upgrade.enabled:
                missing_up = [
                    t for t in ("btn_upgrades", "btn_upgrades_close") if not self.vision.has_template(t)
                ]
                if missing_up:
                    self._emit_log(
                        "WARN",
                        f"Auto-Upgrade is enabled, but missing template(s): {', '.join(missing_up)}. "
                        "Run calibrate.bat to capture them.",
                    )

            self.dry_run = dry_run
            self.stop_event.clear()
            self._stop_reason = None
            self.stats = MacroStats()
            self._recovery_timestamps.clear()
            self._run_end_unconfirmed = False
            self._pending_spawner_pass = bool(
                self.config.auto_spawner.enabled and self.config.auto_spawner.run_on_start
            )
            self._pending_upgrade_pass = bool(
                self.config.auto_upgrade.enabled and self.config.auto_upgrade.run_on_start
            )
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
                    if self._stop_reason:
                        stop_reason = self._stop_reason
                    elif self.stats.circuit_breaker_tripped:
                        stop_reason = "Circuit breaker tripped"
                    break
                if delay > 0:
                    self.stop_event.wait(delay)
        except Exception as exc:
            stop_reason = f"Unhandled FSM error: {exc}"
            self._emit_log("ERROR", stop_reason)
        finally:
            self.close()
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
        if self.state == MacroState.PURCHASING_SPAWNERS:
            return self._step_purchasing_spawners()
        if self.state == MacroState.UPGRADING:
            return self._step_upgrading()
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
        if self.state == MacroState.PURCHASING_SPAWNERS:
            return t.purchasing_spawners_sec
        if self.state == MacroState.UPGRADING:
            return t.upgrading_sec
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
            # Back in a run (HUD reappeared or a run/resume was started): any prior HUD loss
            # was a mid-run dropout, not a completed run.
            self._run_end_unconfirmed = False
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

    def _can_run_auto_spawners(self) -> bool:
        """Return True when auto-spawner is enabled and required Spawner Shop UI templates exist."""
        return bool(
            self.config.auto_spawner.enabled
            and self.vision.has_template("btn_spawners")
            and self.vision.has_template("btn_spawners_close")
        )

    def _can_run_auto_upgrade(self) -> bool:
        """Return True when auto-upgrade is enabled and required UI templates exist."""
        return bool(
            self.config.auto_upgrade.enabled
            and self.vision.has_template("btn_upgrades")
            and self.vision.has_template("btn_upgrades_close")
        )

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
            if self._run_end_unconfirmed:
                self._record_run_completed("Battle button appeared after wave HUD was lost")
            if self._pending_spawner_pass:
                self._pending_spawner_pass = False
                if self._can_run_auto_spawners():
                    self._transition_to(MacroState.PURCHASING_SPAWNERS)
                    return 0.0
            if self._pending_upgrade_pass:
                self._pending_upgrade_pass = False
                if self._can_run_auto_upgrade():
                    self._transition_to(MacroState.UPGRADING)
                    return 0.0
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

        if self.dry_run and self.config.auto_spawner.enabled and self.vision.has_template("btn_spawners_close"):
            sp_close_vis = self.vision.find(
                frame,
                "btn_spawners_close",
                self.config.thresholds.btn_spawners_close,
                self.config.rois.btn_spawners_close,
            )
            if sp_close_vis.found:
                self._transition_to(MacroState.PURCHASING_SPAWNERS)
                return 0.0

        if self.dry_run and self.config.auto_upgrade.enabled and self.vision.has_template("btn_upgrades_close"):
            close_vis = self.vision.find(
                frame,
                "btn_upgrades_close",
                self.config.thresholds.btn_upgrades_close,
                self.config.rois.btn_upgrades_close,
            )
            if close_vis.found:
                self._transition_to(MacroState.UPGRADING)
                return 0.0

        if not self.dry_run:
            # Safety net: a Spawner Shop / Upgrade Tree left open hides Battle and would stall
            # READY_TO_START until the watchdog fires. Close it and re-evaluate next tick.
            for modal_close, modal_label in (
                ("btn_spawners_close", "Spawner Shop"),
                ("btn_upgrades_close", "Upgrade Tree"),
            ):
                if not self.vision.has_template(modal_close):
                    continue
                modal_th = getattr(self.config.thresholds, modal_close)
                modal_roi = getattr(self.config.rois, modal_close)
                if self.vision.find(frame, modal_close, modal_th, modal_roi).found:
                    self._emit_log("WARN", f"{modal_label} left open in READY_TO_START; closing it.")
                    status = self._close_modal_via_template(
                        close_template=modal_close,
                        close_threshold=modal_th,
                        close_roi=modal_roi,
                        disconnect_reason="modal_disconnected_ready",
                    )
                    if status == "stopped":
                        return None
                    return 0.0 if status == "disconnected" else menu_poll

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
        else:
            # Nothing recognised. If another window is covering Roblox, the screen capture can't see the
            # game and the watchdog would relaunch it; bring Roblox back to the front instead (throttled).
            is_focused = getattr(self.controller, "is_focused", None)
            now = self.clock()
            if (
                callable(is_focused)
                and hwnd is not None
                and not is_focused(hwnd)
                and now - self._last_refocus_at >= 5.0
            ):
                self._last_refocus_at = now
                self._emit_log("INFO", "Roblox is not the foreground window; bringing it to the front.")
                self.controller.focus(hwnd)

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

    def _open_modal_via_template(
        self,
        *,
        hwnd: Optional[int],
        frame: np.ndarray,
        open_template: str,
        open_threshold: float,
        open_roi: list[float],
        close_template: str,
        close_threshold: float,
        close_roi: list[float],
        disconnect_reason: str,
        state_label: str,
    ) -> tuple[str, Optional[int], Optional[np.ndarray]]:
        """Ensure a modal is open via close_template, clicking open_template if needed."""
        th = self.config.thresholds
        rois = self.config.rois
        polling = self.config.polling

        if self.vision.find(frame, close_template, close_threshold, close_roi).found:
            return ("ok", hwnd, frame)

        open_btn = self.vision.find(frame, open_template, open_threshold, open_roi)
        if not open_btn.found or open_btn.center is None:
            self._emit_log("WARN", f"{state_label} skipped: {open_template} not visible in lobby.")
            return ("skip", hwnd, frame)

        confirm_timeout = max(0.001, float(polling.click_confirm_timeout_sec))
        poll_sec = max(0.001, float(polling.click_confirm_poll_sec))
        max_polls = max(1, math.ceil(confirm_timeout / poll_sec))
        open_attempts = 3
        for attempt in range(open_attempts):
            # Roblox ignores clicks that land without a prior hover (e.g. right after it regains focus).
            self.controller.hover_client_point(open_btn.center[0], open_btn.center[1], hwnd=hwnd, dry_run=False)
            if self.stop_event.wait(0.12):
                return ("stopped", None, None)
            if not self.controller.click_client_point(
                open_btn.center[0],
                open_btn.center[1],
                hwnd=hwnd,
                dry_run=False,
            ):
                self._emit_log("WARN", f"Failed to focus/click {open_template}; returning to READY_TO_START.")
                return ("skip", hwnd, frame)

            deadline = self.clock() + confirm_timeout
            polls = 0
            while True:
                if self.stop_event.wait(poll_sec):
                    return ("stopped", None, None)
                polls += 1
                hwnd_open, open_frame = self._capture_current_frame()
                if open_frame is not None:
                    disc_open = self.vision.find(
                        open_frame,
                        "modal_disconnected",
                        th.modal_disconnected,
                        rois.modal_disconnected,
                    )
                    if disc_open.found:
                        self._disconnect_reason = disconnect_reason
                        self._transition_to(MacroState.DISCONNECTED)
                        return ("disconnected", hwnd_open, None)
                    if self.vision.find(open_frame, close_template, close_threshold, close_roi).found:
                        return ("ok", hwnd_open, open_frame)
                if polls >= max_polls or self.clock() >= deadline:
                    break

            if attempt + 1 < open_attempts:
                self._emit_log(
                    "WARN",
                    f"{state_label}: {open_template} click did not open it (attempt {attempt + 1}/{open_attempts}); retrying.",
                )
                hwnd, retry_frame = self._capture_current_frame()
                if retry_frame is not None:
                    frame = retry_frame
                    retry_btn = self.vision.find(frame, open_template, open_threshold, open_roi)
                    if retry_btn.found and retry_btn.center is not None:
                        open_btn = retry_btn

        self._emit_log("WARN", f"{state_label} did not open after clicking {open_template}; returning to READY_TO_START.")
        return ("skip", hwnd, frame)

    def _confirm_modal_absent(
        self,
        close_template: str,
        close_threshold: float,
        close_roi: list[float],
        polls: int = 4,
    ) -> tuple[bool, Optional[int], Optional[np.ndarray]]:
        """Re-check a modal's close button over a few frames.

        Returns (True, hwnd, None) if it stays missing on every poll, or (False, hwnd, frame)
        with the first frame where it is visible again.
        """
        poll = max(0.001, float(self.config.polling.click_confirm_poll_sec))
        hwnd: Optional[int] = None
        for _ in range(max(1, polls)):
            if self.stop_event.wait(poll):
                return (True, hwnd, None)
            hwnd, frame = self._capture_current_frame()
            if frame is None:
                return (True, hwnd, None)
            if self.vision.find(frame, close_template, close_threshold, close_roi).found:
                return (False, hwnd, frame)
        return (True, hwnd, None)

    def _close_modal_via_template(
        self,
        *,
        close_template: str,
        close_threshold: float,
        close_roi: list[float],
        disconnect_reason: str,
    ) -> str:
        """Close an open modal via close_template (up to 3 attempts with Esc fallback)."""
        th = self.config.thresholds
        rois = self.config.rois
        polling = self.config.polling

        close_timeout = max(0.001, float(polling.click_confirm_timeout_sec))
        close_poll = max(0.001, float(polling.click_confirm_poll_sec))
        close_max_polls = max(1, math.ceil(close_timeout / close_poll))
        for attempt in range(3):
            if self.stop_event.is_set():
                return "stopped"
            hwnd_close, close_frame = self._capture_current_frame()
            if close_frame is None:
                return "ok"
            disc_close = self.vision.find(
                close_frame,
                "modal_disconnected",
                th.modal_disconnected,
                rois.modal_disconnected,
            )
            if disc_close.found:
                self._disconnect_reason = disconnect_reason
                self._transition_to(MacroState.DISCONNECTED)
                return "disconnected"

            close_vis = self.vision.find(close_frame, close_template, close_threshold, close_roi)
            if not close_vis.found:
                absent, re_hwnd, re_frame = self._confirm_modal_absent(
                    close_template, close_threshold, close_roi
                )
                if self.stop_event.is_set():
                    return "stopped"
                if absent or re_frame is None:
                    return "ok"
                hwnd_close = re_hwnd
                close_vis = self.vision.find(re_frame, close_template, close_threshold, close_roi)

            if close_vis.center is not None:
                self.controller.click_client_point(
                    close_vis.center[0],
                    close_vis.center[1],
                    hwnd=hwnd_close,
                    dry_run=False,
                )

            closed_ok = False
            close_deadline = self.clock() + close_timeout
            for _ in range(close_max_polls):
                if self.stop_event.wait(close_poll):
                    return "stopped"
                _, check_frame = self._capture_current_frame()
                if check_frame is None:
                    closed_ok = True
                    break
                if not self.vision.find(check_frame, close_template, close_threshold, close_roi).found:
                    closed_ok = True
                    break
                if self.clock() >= close_deadline:
                    break

            if closed_ok:
                return "ok"

            self._emit_log(
                "WARN",
                f"{close_template} still visible after click (attempt {attempt + 1}/3); sending Esc fallback.",
            )
            self.controller.send_key("esc", hwnd=hwnd_close, dry_run=False)
            if self.stop_event.wait(close_poll):
                return "stopped"

        return "ok"

    def _step_purchasing_spawners(self) -> Optional[float]:
        if self.stop_event.is_set():
            return None

        cfg = self.config.auto_spawner
        th = self.config.thresholds
        rois = self.config.rois
        polling = self.config.polling

        hwnd, frame = self._capture_current_frame()
        if frame is None:
            if self.dry_run:
                self._emit_log("WARN", "Dry-Run (PURCHASING_SPAWNERS): unable to capture Roblox client frame.")
                self._transition_to(MacroState.READY_TO_START)
                return polling.menu_interval_sec
            if not self.controller.is_roblox_running():
                self._disconnect_reason = "process_exited_spawners"
                self._transition_to(MacroState.DISCONNECTED)
                return 0.0
            self._emit_log("WARN", "Unable to capture frame in PURCHASING_SPAWNERS; returning to READY_TO_START.")
            self._transition_to(MacroState.READY_TO_START)
            return polling.menu_interval_sec

        disc = self.vision.find(frame, "modal_disconnected", th.modal_disconnected, rois.modal_disconnected)
        if disc.found:
            if self.dry_run:
                self._emit_log("INFO", "Dry-Run (PURCHASING_SPAWNERS): disconnect modal visible.")
                self._transition_to(MacroState.READY_TO_START)
                return polling.menu_interval_sec
            self._disconnect_reason = "modal_disconnected_spawners"
            self._transition_to(MacroState.DISCONNECTED)
            return 0.0

        if self.dry_run:
            close_vis = self.vision.find(frame, "btn_spawners_close", th.btn_spawners_close, rois.btn_spawners_close)
            if close_vis.found:
                scan = self.spawner_scanner(frame)
                coords = [b.center for b in scan.buttons]
                self._emit_log(
                    "INFO",
                    f"Dry-Run (PURCHASING_SPAWNERS): detected {len(scan.buttons)} green Buy pill(s): {coords}",
                )
            else:
                sp_btn = self.vision.find(frame, "btn_spawners", th.btn_spawners, rois.btn_spawners)
                self._emit_log(
                    "INFO",
                    f"Dry-Run (PURCHASING_SPAWNERS): shop closed; would open via btn_spawners "
                    f"(found={sp_btn.found}, conf={sp_btn.confidence:.3f}) without clicking.",
                )
            self._transition_to(MacroState.READY_TO_START)
            return polling.menu_interval_sec

        open_status, hwnd, frame = self._open_modal_via_template(
            hwnd=hwnd,
            frame=frame,
            open_template="btn_spawners",
            open_threshold=th.btn_spawners,
            open_roi=rois.btn_spawners,
            close_template="btn_spawners_close",
            close_threshold=th.btn_spawners_close,
            close_roi=rois.btn_spawners_close,
            disconnect_reason="modal_disconnected_spawners",
            state_label="Auto-Spawners",
        )
        if open_status == "stopped":
            return None
        if open_status == "disconnected":
            return 0.0
        if open_status != "ok" or frame is None:
            self._transition_to(MacroState.READY_TO_START)
            return 0.0

        pass_deadline = self.clock() + max(0.0, float(cfg.pass_timeout_sec))
        max_clicks = max(0, int(cfg.max_purchases_per_pass))
        max_scrolls = max(0, int(cfg.max_scroll_steps))
        scroll_notches = max(1, abs(int(cfg.scroll_notches_per_step)))
        rounds_per_page = max(1, int(cfg.clicks_per_pill))
        click_interval = max(0.0, float(cfg.click_interval_sec))
        round_settle = max(0.0, float(cfg.round_settle_sec))
        hover_sec = max(0.0, float(cfg.click_hover_sec))
        clicks_in_pass = 0
        pages_visited = 0
        unchanged_scroll_streak = 0
        empty_page_streak = 0
        stop_reason = "end of list"

        def _capture_and_scan_spawners() -> tuple[str, Optional[int], Optional[np.ndarray], Optional[SpawnerScanResult]]:
            if self.stop_event.is_set():
                return ("stopped", None, None, None)
            cur_hwnd, cur_frame = self._capture_current_frame()
            if cur_frame is None:
                if not self.controller.is_roblox_running():
                    self._disconnect_reason = "process_exited_spawners"
                    self._transition_to(MacroState.DISCONNECTED)
                    return ("disconnected", cur_hwnd, None, None)
                return ("shop_closed", cur_hwnd, None, None)

            cur_disc = self.vision.find(
                cur_frame,
                "modal_disconnected",
                th.modal_disconnected,
                rois.modal_disconnected,
            )
            if cur_disc.found:
                self._disconnect_reason = "modal_disconnected_spawners"
                self._transition_to(MacroState.DISCONNECTED)
                return ("disconnected", cur_hwnd, None, None)

            if not self.vision.find(
                cur_frame,
                "btn_spawners_close",
                th.btn_spawners_close,
                rois.btn_spawners_close,
            ).found:
                # A purchase overlay can briefly hide the close X; only treat the shop as
                # closed if the X stays missing across several frames.
                absent, re_hwnd, re_frame = self._confirm_modal_absent(
                    "btn_spawners_close", th.btn_spawners_close, rois.btn_spawners_close
                )
                if self.stop_event.is_set():
                    return ("stopped", cur_hwnd, None, None)
                if absent or re_frame is None:
                    return ("shop_closed", cur_hwnd, None, None)
                cur_hwnd, cur_frame = re_hwnd, re_frame

            return ("ok", cur_hwnd, cur_frame, self.spawner_scanner(cur_frame))

        def _pills_cheapest_first(scan: SpawnerScanResult) -> list[SpawnerBuyButton]:
            # The shop is laid out cheapest -> most expensive: top row left-to-right, then the next row.
            return sorted(scan.buttons, key=lambda b: (b.row_index, b.col_index, b.center[1], b.center[0]))

        cur_hwnd = hwnd
        cur_frame = frame
        cur_scan = self.spawner_scanner(cur_frame)
        finished = False

        # Simple and lag-proof: click every green Buy pill on the page, rescan, and repeat until no green
        # pills remain (multi-stock items stay green until sold out) or the per-page round cap is hit.
        # No OCR or purchase confirmation: Sold out / Robux pills are never green, so they are never clicked.
        for scroll_step in range(max_scrolls + 1):
            pages_visited += 1
            pills_left_after_cap = False
            for round_idx in range(rounds_per_page):
                pills = _pills_cheapest_first(cur_scan)
                if not pills:
                    break
                for pill in pills:
                    if self.stop_event.is_set():
                        return None
                    if clicks_in_pass >= max_clicks:
                        stop_reason = "click limit reached"
                        finished = True
                        break
                    if self.clock() >= pass_deadline:
                        stop_reason = "pass timeout"
                        finished = True
                        break
                    # Roblox ignores clicks that land without a brief hover first.
                    self.controller.hover_client_point(pill.center[0], pill.center[1], hwnd=cur_hwnd, dry_run=False)
                    if hover_sec > 0 and self.stop_event.wait(hover_sec):
                        return None
                    self.controller.click_client_point(
                        pill.center[0],
                        pill.center[1],
                        hwnd=cur_hwnd,
                        dry_run=False,
                        park_cursor=False,
                    )
                    clicks_in_pass += 1
                    self.stats.spawners_purchased += 1
                    if click_interval > 0 and self.stop_event.wait(click_interval):
                        return None
                self._emit_metrics()
                if finished:
                    break
                # Give the server time to apply the purchases (high ping) and the coin bursts time to clear.
                self.controller.park_cursor_neutral(hwnd=cur_hwnd, dry_run=False)
                if round_settle > 0 and self.stop_event.wait(round_settle):
                    return None
                status, cur_hwnd, new_frame, new_scan = _capture_and_scan_spawners()
                if status == "stopped":
                    return None
                if status == "disconnected":
                    return 0.0
                if status != "ok" or new_frame is None or new_scan is None:
                    stop_reason = "shop closed unexpectedly"
                    finished = True
                    break
                cur_frame, cur_scan = new_frame, new_scan
                if round_idx == rounds_per_page - 1 and cur_scan.buttons:
                    pills_left_after_cap = True

            if finished or scroll_step >= max_scrolls:
                break
            if pills_left_after_cap:
                # Still green after every round: most likely out of cash. Later pages only cost more.
                stop_reason = "Buy buttons still green after max rounds (likely out of cash)"
                break

            gx1, gy1, gx2, gy2 = cur_scan.grid_roi_px
            if gx2 > gx1 and gy2 > gy1:
                scroll_x = (gx1 + gx2) // 2
                scroll_y = (gy1 + gy2) // 2
            else:
                rect = self.controller.get_client_rect(cur_hwnd)
                cw, ch = (rect[2], rect[3]) if rect is not None else (1280, 720)
                scroll_x, scroll_y = cw // 2, int(ch * 0.56)

            prev_scroll_frame = cur_frame
            self.controller.scroll_client_point(
                scroll_x,
                scroll_y,
                -scroll_notches,
                hwnd=cur_hwnd,
                dry_run=False,
            )
            if round_settle > 0 and self.stop_event.wait(min(round_settle, 0.6)):
                return None

            status, cur_hwnd, post_scroll_frame, post_scroll_scan = _capture_and_scan_spawners()
            if status == "stopped":
                return None
            if status == "disconnected":
                return 0.0
            if status != "ok" or post_scroll_frame is None or post_scroll_scan is None:
                stop_reason = "shop closed unexpectedly"
                break

            if is_scroll_grid_unchanged(prev_scroll_frame, post_scroll_frame) and not post_scroll_scan.buttons:
                unchanged_scroll_streak += 1
                if unchanged_scroll_streak >= 2:
                    break
            else:
                unchanged_scroll_streak = 0
            # Animated cards can keep the grid "changing" even at the bottom of the list, so also stop after
            # several scroll steps in a row that show no green Buy pills at all.
            empty_page_streak = empty_page_streak + 1 if not post_scroll_scan.buttons else 0
            if empty_page_streak >= 3:
                break

            cur_frame = post_scroll_frame
            cur_scan = post_scroll_scan

        close_status = self._close_modal_via_template(
            close_template="btn_spawners_close",
            close_threshold=th.btn_spawners_close,
            close_roi=rois.btn_spawners_close,
            disconnect_reason="modal_disconnected_spawners",
        )
        if close_status == "stopped":
            return None
        if close_status == "disconnected":
            return 0.0

        self._emit_log(
            "INFO",
            f"Auto-Spawners pass complete: clicked Buy {clicks_in_pass} time(s) across {pages_visited} page(s) "
            f"({stop_reason}; session total: {self.stats.spawners_purchased} Buy clicks).",
        )
        self._transition_to(MacroState.READY_TO_START)
        return 0.0

    def _step_upgrading(self) -> Optional[float]:
        if self.stop_event.is_set():
            return None

        cfg = self.config.auto_upgrade
        th = self.config.thresholds
        rois = self.config.rois
        polling = self.config.polling

        hwnd, frame = self._capture_current_frame()
        if frame is None:
            if self.dry_run:
                self._emit_log("WARN", "Dry-Run (UPGRADING): unable to capture Roblox client frame.")
                self._transition_to(MacroState.READY_TO_START)
                return polling.menu_interval_sec
            if not self.controller.is_roblox_running():
                self._disconnect_reason = "process_exited_upgrading"
                self._transition_to(MacroState.DISCONNECTED)
                return 0.0
            self._emit_log("WARN", "Unable to capture frame in UPGRADING; returning to READY_TO_START.")
            self._transition_to(MacroState.READY_TO_START)
            return polling.menu_interval_sec

        disc = self.vision.find(frame, "modal_disconnected", th.modal_disconnected, rois.modal_disconnected)
        if disc.found:
            if self.dry_run:
                self._emit_log("INFO", "Dry-Run (UPGRADING): disconnect modal visible.")
                self._transition_to(MacroState.READY_TO_START)
                return polling.menu_interval_sec
            self._disconnect_reason = "modal_disconnected_upgrading"
            self._transition_to(MacroState.DISCONNECTED)
            return 0.0

        if self.dry_run:
            close_vis = self.vision.find(frame, "btn_upgrades_close", th.btn_upgrades_close, rois.btn_upgrades_close)
            if close_vis.found:
                scan = self.upgrade_scanner(
                    frame,
                    scope=cfg.scope,
                    order=cfg.order,
                    tile_click_y_offset_px=cfg.tile_click_y_offset_px,
                )
                main_cands = [c for c in scan.all_candidates if c.is_main_tree]
                side_cands = [c for c in scan.all_candidates if not c.is_main_tree]
                coords = [c.click_point for c in scan.candidates]
                edges = ",".join(sorted(scan.clipped_edges)) if scan.clipped_edges else "none"
                start_h = scan.start_anchor.height_px if scan.start_anchor else "none"
                self._emit_log(
                    "INFO",
                    f"Dry-Run (UPGRADING): {len(main_cands)} main-tree and {len(side_cands)} side-island "
                    f"affordable tile(s) (scope={cfg.scope}, start_h={start_h}, cov={scan.tree_coverage_ratio:.2f}, "
                    f"clipped={edges}): {coords}",
                )
            else:
                up_btn = self.vision.find(frame, "btn_upgrades", th.btn_upgrades, rois.btn_upgrades)
                self._emit_log(
                    "INFO",
                    f"Dry-Run (UPGRADING): tree closed; would open via btn_upgrades "
                    f"(found={up_btn.found}, conf={up_btn.confidence:.3f}) without clicking.",
                )
            self._transition_to(MacroState.READY_TO_START)
            return polling.menu_interval_sec

        # 1. Open the upgrade tree if btn_upgrades_close is not already visible
        open_status, hwnd, frame = self._open_modal_via_template(
            hwnd=hwnd,
            frame=frame,
            open_template="btn_upgrades",
            open_threshold=th.btn_upgrades,
            open_roi=rois.btn_upgrades,
            close_template="btn_upgrades_close",
            close_threshold=th.btn_upgrades_close,
            close_roi=rois.btn_upgrades_close,
            disconnect_reason="modal_disconnected_upgrading",
            state_label="Auto-Upgrade",
        )
        if open_status == "stopped":
            return None
        if open_status == "disconnected":
            return 0.0
        if open_status != "ok" or frame is None:
            self._transition_to(MacroState.READY_TO_START)
            return 0.0

        rect = self.controller.get_client_rect(hwnd)
        client_w, client_h = (rect[2], rect[3]) if rect is not None else (1280, 720)
        center_x, center_y = client_w // 2, client_h // 2
        zoom_settle = min(0.25, max(0.0, float(cfg.post_click_settle_sec)))
        pass_deadline = self.clock() + max(0.0, float(cfg.pass_timeout_sec))
        max_purchases = max(0, int(cfg.max_purchases_per_pass))
        purchases_in_pass = 0

        def _capture_and_scan() -> tuple[str, Optional[int], Optional[np.ndarray], Optional[TreeScanResult]]:
            if self.stop_event.is_set():
                return ("stopped", None, None, None)
            cur_hwnd, cur_frame = self._capture_current_frame()
            if cur_frame is None:
                if not self.controller.is_roblox_running():
                    self._disconnect_reason = "process_exited_upgrading"
                    self._transition_to(MacroState.DISCONNECTED)
                    return ("disconnected", cur_hwnd, None, None)
                return ("tree_closed", cur_hwnd, None, None)

            cur_disc = self.vision.find(
                cur_frame,
                "modal_disconnected",
                th.modal_disconnected,
                rois.modal_disconnected,
            )
            if cur_disc.found:
                self._disconnect_reason = "modal_disconnected_upgrading"
                self._transition_to(MacroState.DISCONNECTED)
                return ("disconnected", cur_hwnd, None, None)

            if not self.vision.find(
                cur_frame,
                "btn_upgrades_close",
                th.btn_upgrades_close,
                rois.btn_upgrades_close,
            ).found:
                return ("tree_closed", cur_hwnd, None, None)

            scan_res = self.upgrade_scanner(
                cur_frame,
                scope=cfg.scope,
                order=cfg.order,
                tile_click_y_offset_px=cfg.tile_click_y_offset_px,
            )
            return ("ok", cur_hwnd, cur_frame, scan_res)

        def _recenter_on_start(cur_hwnd: Optional[int], cur_frame: np.ndarray, cur_scan: TreeScanResult) -> tuple[str, Optional[int], Optional[np.ndarray], Optional[TreeScanResult]]:
            fh, fw = cur_frame.shape[:2]
            scale = min(fw / 1280.0, fh / 720.0)
            tol_px = max(12.0, float(cfg.center_tolerance_px) * scale)
            hud_mask = build_hud_mask(fh, fw)
            target_x, target_y = fw // 2, fh // 2

            for _ in range(3):
                if cur_scan.start_anchor is None:
                    break
                sx, sy = cur_scan.start_anchor.center
                dx = target_x - sx
                dy = target_y - sy
                if math.hypot(dx, dy) <= tol_px:
                    break

                drag_pair = find_empty_drag_origin(
                    cur_frame.shape,
                    cur_scan.main_tree_bbox,
                    hud_mask,
                    dx,
                    dy,
                )
                if drag_pair is None:
                    # Clamp drag vector if full offset wouldn't fit inside empty margin
                    clamped_dx = max(-fw // 4, min(fw // 4, dx))
                    clamped_dy = max(-fh // 4, min(fh // 4, dy))
                    drag_pair = find_empty_drag_origin(
                        cur_frame.shape,
                        cur_scan.main_tree_bbox,
                        hud_mask,
                        clamped_dx,
                        clamped_dy,
                    )
                if drag_pair is None:
                    break

                (x1, y1), (x2, y2) = drag_pair
                if not self.controller.drag_client_points(x1, y1, x2, y2, hwnd=cur_hwnd, dry_run=False):
                    break
                self.controller.park_cursor_neutral(hwnd=cur_hwnd, dry_run=False)
                if zoom_settle > 0 and self.stop_event.wait(zoom_settle):
                    return ("stopped", None, None, None)

                st, cur_hwnd, next_frame, next_scan = _capture_and_scan()
                if st != "ok" or next_frame is None or next_scan is None:
                    return (st, cur_hwnd, next_frame, next_scan)
                cur_frame, cur_scan = next_frame, next_scan

            return ("ok", cur_hwnd, cur_frame, cur_scan)

        # 2. Visual Start-anchor recovery, closed-loop centering, and adaptive zoom
        if cfg.adaptive_zoom:
            self.controller.park_cursor_neutral(hwnd=hwnd, dry_run=False)
            status, hwnd, cur_frame, cur_scan = _capture_and_scan()
            if status in ("stopped", "disconnected"):
                return None if status == "stopped" else 0.0

            if status == "ok" and cur_frame is not None and cur_scan is not None:
                # Recover if camera started in blank void (Start anchor not visible)
                if cur_scan.start_anchor is None:
                    self.controller.scroll_client_point(center_x, center_y, -8, hwnd=hwnd, dry_run=False)
                    self.controller.park_cursor_neutral(hwnd=hwnd, dry_run=False)
                    if zoom_settle > 0 and self.stop_event.wait(zoom_settle):
                        return None
                    status, hwnd, cur_frame, cur_scan = _capture_and_scan()
                    if status in ("stopped", "disconnected"):
                        return None if status == "stopped" else 0.0

                if status == "ok" and cur_scan is not None and cur_scan.start_anchor is None:
                    self._emit_log("WARN", "Upgrade tree Start anchor missing; closing and reopening tree once to reset view.")
                    close_rst = self._close_modal_via_template(
                        close_template="btn_upgrades_close",
                        close_threshold=th.btn_upgrades_close,
                        close_roi=rois.btn_upgrades_close,
                        disconnect_reason="modal_disconnected_upgrading",
                    )
                    if close_rst == "stopped":
                        return None
                    if close_rst == "disconnected":
                        return 0.0

                    hwnd_rst, frame_rst = self._capture_current_frame()
                    if frame_rst is not None:
                        reopen_status, hwnd, reopened_frame = self._open_modal_via_template(
                            hwnd=hwnd_rst,
                            frame=frame_rst,
                            open_template="btn_upgrades",
                            open_threshold=th.btn_upgrades,
                            open_roi=rois.btn_upgrades,
                            close_template="btn_upgrades_close",
                            close_threshold=th.btn_upgrades_close,
                            close_roi=rois.btn_upgrades_close,
                            disconnect_reason="modal_disconnected_upgrading",
                            state_label="Auto-Upgrade",
                        )
                        if reopen_status == "stopped":
                            return None
                        if reopen_status == "disconnected":
                            return 0.0
                        if reopen_status == "ok":
                            self.controller.park_cursor_neutral(hwnd=hwnd, dry_run=False)
                            status, hwnd, cur_frame, cur_scan = _capture_and_scan()
                            if status in ("stopped", "disconnected"):
                                return None if status == "stopped" else 0.0

                # Adaptive zoom toward Start tile target height while avoiding unnecessary edge clipping
                if status == "ok" and cur_frame is not None and cur_scan is not None and cur_scan.start_anchor is not None:
                    fh, fw = cur_frame.shape[:2]
                    scale = min(fw / 1280.0, fh / 720.0)
                    min_h = max(12.0, float(cfg.min_start_tile_height_px) * scale)
                    max_h = max(min_h + 4.0, float(cfg.max_start_tile_height_px) * scale)

                    for _ in range(5):
                        if cur_scan.start_anchor is None:
                            break
                        sh = float(cur_scan.start_anchor.height_px)
                        notches = 0
                        if sh > max_h or (cur_scan.clipped_edges and sh >= min_h + 2.0):
                            notches = -2 if sh > max_h * 1.4 else -1
                        elif sh < min_h and not cur_scan.clipped_edges:
                            notches = 2 if sh < min_h * 0.7 else 1

                        if notches == 0:
                            break

                        zx, zy = cur_scan.start_anchor.center
                        self.controller.scroll_client_point(zx, zy, notches, hwnd=hwnd, dry_run=False)
                        self.controller.park_cursor_neutral(hwnd=hwnd, dry_run=False)
                        if zoom_settle > 0 and self.stop_event.wait(zoom_settle):
                            return None
                        status, hwnd, cur_frame, cur_scan = _capture_and_scan()
                        if status in ("stopped", "disconnected"):
                            return None if status == "stopped" else 0.0
                        if status != "ok" or cur_frame is None or cur_scan is None:
                            break

                # Center visually onto Start using verified empty-background drags
                if status == "ok" and cur_frame is not None and cur_scan is not None and cur_scan.start_anchor is not None:
                    status, hwnd, cur_frame, cur_scan = _recenter_on_start(hwnd, cur_frame, cur_scan)
                    if status in ("stopped", "disconnected"):
                        return None if status == "stopped" else 0.0
        else:
            # Legacy fixed-scroll zoom path when adaptive_zoom is explicitly disabled
            if int(cfg.zoom_reset_scrolls) > 0:
                self.controller.scroll_client_point(
                    center_x,
                    center_y,
                    -abs(int(cfg.zoom_reset_scrolls)),
                    hwnd=hwnd,
                    dry_run=False,
                )
            if self.stop_event.is_set():
                return None
            if int(cfg.zoom_in_scrolls) > 0:
                self.controller.scroll_client_point(
                    center_x,
                    center_y,
                    abs(int(cfg.zoom_in_scrolls)),
                    hwnd=hwnd,
                    dry_run=False,
                )
            if self.stop_event.is_set():
                return None
            self.controller.park_cursor_neutral(hwnd=hwnd, dry_run=False)
            if zoom_settle > 0 and self.stop_event.wait(zoom_settle):
                return None
            status, hwnd, cur_frame, cur_scan = _capture_and_scan()
            if status in ("stopped", "disconnected"):
                return None if status == "stopped" else 0.0

        # 3. Scan & hover-gated purchase loop helper
        def _upgrade_sector(
            sector_name: str,
            initial_hwnd: Optional[int],
            initial_frame: Optional[np.ndarray],
            initial_scan: Optional[TreeScanResult],
        ) -> tuple[str, set[str], Optional[int], Optional[np.ndarray], Optional[TreeScanResult]]:
            nonlocal purchases_in_pass
            cur_h, cur_f, scan = initial_hwnd, initial_frame, initial_scan
            if cur_f is None or scan is None:
                st, cur_h, cur_f, scan = _capture_and_scan()
                if st != "ok" or cur_f is None or scan is None:
                    return (st, set(), cur_h, cur_f, scan)

            sector_clipped = set(scan.clipped_edges)
            blocked_or_failed_points: list[tuple[int, int]] = []
            retry_entries: list[list[Any]] = []

            def _is_blacklisted(pt: tuple[int, int]) -> bool:
                return any(is_same_candidate(pt, fp) for fp in blocked_or_failed_points)

            def _record_retry(pt: tuple[int, int]) -> int:
                for entry in retry_entries:
                    if is_same_candidate(pt, entry[0]):
                        entry[1] += 1
                        return int(entry[1])
                retry_entries.append([pt, 1])
                return 1

            while True:
                if self.stop_event.is_set():
                    return ("stopped", sector_clipped, cur_h, cur_f, scan)
                if purchases_in_pass >= max_purchases or self.clock() >= pass_deadline:
                    return ("budget_exhausted", sector_clipped, cur_h, cur_f, scan)

                target: Optional[UpgradeCandidate] = next(
                    (c for c in scan.candidates if not _is_blacklisted(c.click_point)),
                    None,
                )
                if target is None:
                    return ("done", sector_clipped, cur_h, cur_f, scan)

                # Mandatory hover-before-click Board guard
                hovered = self.controller.hover_client_point(
                    target.click_point[0],
                    target.click_point[1],
                    hwnd=cur_h,
                    dry_run=False,
                )
                hover_settle = max(0.0, float(cfg.hover_settle_sec))
                if hover_settle > 0 and self.stop_event.wait(hover_settle):
                    return ("stopped", sector_clipped, cur_h, cur_f, scan)

                hover_hwnd, hover_frame = self._capture_current_frame()
                if hover_frame is None:
                    if not self.controller.is_roblox_running():
                        self._disconnect_reason = "process_exited_upgrading"
                        self._transition_to(MacroState.DISCONNECTED)
                        return ("disconnected", sector_clipped, hover_hwnd, None, None)
                    return ("tree_closed", sector_clipped, hover_hwnd, None, None)

                hover_disc = self.vision.find(
                    hover_frame,
                    "modal_disconnected",
                    th.modal_disconnected,
                    rois.modal_disconnected,
                )
                if hover_disc.found:
                    self._disconnect_reason = "modal_disconnected_upgrading"
                    self._transition_to(MacroState.DISCONNECTED)
                    return ("disconnected", sector_clipped, hover_hwnd, None, None)

                tooltip_crop = self.tooltip_extractor(cur_f, hover_frame, target.click_point) if hovered else None
                ocr_lines = self._get_ocr_lines(tooltip_crop, invert=True, scale=3.0) if tooltip_crop is not None else []
                verdict = classify_upgrade_tooltip(ocr_lines, cfg.blocked_keywords)

                if verdict.status == "blocked":
                    blocked_or_failed_points.append(target.click_point)
                    self.controller.park_cursor_neutral(hwnd=hover_hwnd, dry_run=False)
                    self._emit_log(
                        "INFO",
                        f'Blocked upgrade "{verdict.title or "unknown"}" at {target.click_point} ({verdict.reason}).',
                    )
                    continue

                if verdict.status == "unlocked":
                    blocked_or_failed_points.append(target.click_point)
                    self.controller.park_cursor_neutral(hwnd=hover_hwnd, dry_run=False)
                    self._emit_log(
                        "INFO",
                        f'Skipped already unlocked upgrade "{verdict.title or "unknown"}" at {target.click_point}.',
                    )
                    continue

                if verdict.status == "unreadable" and cfg.fail_closed_tooltip_check:
                    self.controller.park_cursor_neutral(hwnd=hover_hwnd, dry_run=False)
                    tries = _record_retry(target.click_point)
                    if tries >= 2:
                        blocked_or_failed_points.append(target.click_point)
                        self._emit_log(
                            "WARN",
                            f"[{sector_name}] Unreadable tooltip at {target.click_point} twice; fail-closed skip.",
                        )
                    else:
                        if zoom_settle > 0 and self.stop_event.wait(min(0.1, zoom_settle)):
                            return ("stopped", sector_clipped, hover_hwnd, cur_f, scan)
                    continue

                upgrade_name = verdict.title or f"tile@{target.click_point}"
                clicked = self.controller.click_client_point(
                    target.click_point[0],
                    target.click_point[1],
                    hwnd=hover_hwnd,
                    dry_run=False,
                    park_cursor=False,
                )
                self.controller.park_cursor_neutral(hwnd=hover_hwnd, dry_run=False)

                settle = max(0.0, float(cfg.post_click_settle_sec))
                if settle > 0 and self.stop_event.wait(settle):
                    return ("stopped", sector_clipped, hover_hwnd, cur_f, scan)

                st, cur_h, new_frame, new_scan = _capture_and_scan()
                if st != "ok" or new_frame is None or new_scan is None:
                    return (st, sector_clipped, cur_h, new_frame, new_scan)

                if not clicked:
                    tries = _record_retry(target.click_point)
                    if tries >= 2:
                        blocked_or_failed_points.append(target.click_point)
                        self._emit_log(
                            "WARN",
                            f"[{sector_name}] Failed to click upgrade \"{upgrade_name}\" at {target.click_point} twice; blacklisting.",
                        )
                    cur_f, scan = new_frame, new_scan
                    continue

                still_present = any(is_same_candidate(target.click_point, c.click_point) for c in new_scan.candidates)
                if not still_present:
                    purchases_in_pass += 1
                    self.stats.upgrades_purchased += 1
                    self._emit_metrics()
                    self._emit_log(
                        "INFO",
                        f'Purchased upgrade "{upgrade_name}" ({sector_name}, {"main" if target.is_main_tree else "side"}) '
                        f"at {target.click_point} (pass #{purchases_in_pass}, session #{self.stats.upgrades_purchased}).",
                    )
                else:
                    tries = _record_retry(target.click_point)
                    if tries >= 2:
                        blocked_or_failed_points.append(target.click_point)
                        self._emit_log(
                            "WARN",
                            f'Upgrade "{upgrade_name}" at {target.click_point} remained affordable after {tries} clicks; blacklisting.',
                        )

                cur_f, scan = new_frame, new_scan

        if status == "ok":
            status, center_clipped, hwnd, cur_frame, cur_scan = _upgrade_sector("center", hwnd, cur_frame, cur_scan)
            if status in ("stopped", "disconnected"):
                return None if status == "stopped" else 0.0
        else:
            center_clipped = set()

        # 4. Zero-pan fast path when not clipped; guarded empty-background edge panning otherwise
        if (
            status == "done"
            and cfg.enable_edge_panning
            and center_clipped
            and cur_frame is not None
            and cur_scan is not None
            and purchases_in_pass < max_purchases
            and self.clock() < pass_deadline
        ):
            fh, fw = cur_frame.shape[:2]
            hud_mask = build_hud_mask(fh, fw)
            pan_px = max(40, abs(int(cfg.pan_drag_px)))
            pan_deltas: dict[str, tuple[int, int]] = {
                "top": (0, pan_px),
                "bottom": (0, -pan_px),
                "left": (pan_px, 0),
                "right": (-pan_px, 0),
            }
            for direction in ("top", "bottom", "left", "right"):
                if direction not in center_clipped:
                    continue
                if self.stop_event.is_set():
                    return None
                if purchases_in_pass >= max_purchases or self.clock() >= pass_deadline:
                    break

                dx, dy = pan_deltas[direction]
                drag_pair = find_empty_drag_origin(
                    cur_frame.shape,
                    cur_scan.main_tree_bbox,
                    hud_mask,
                    dx,
                    dy,
                )
                if drag_pair is None:
                    continue

                (x1, y1), (x2, y2) = drag_pair
                if not self.controller.drag_client_points(x1, y1, x2, y2, hwnd=hwnd, dry_run=False):
                    continue
                self.controller.park_cursor_neutral(hwnd=hwnd, dry_run=False)
                if zoom_settle > 0 and self.stop_event.wait(zoom_settle):
                    return None

                pan_st, hwnd, pan_frame, pan_scan = _capture_and_scan()
                if pan_st in ("stopped", "disconnected"):
                    return None if pan_st == "stopped" else 0.0

                pan_status = pan_st
                if pan_st == "ok" and pan_frame is not None and pan_scan is not None:
                    if pan_scan.start_anchor is not None or pan_scan.tree_coverage_ratio >= 0.10:
                        pan_status, _, hwnd, pan_frame, pan_scan = _upgrade_sector(
                            f"pan-{direction}",
                            hwnd,
                            pan_frame,
                            pan_scan,
                        )
                        if pan_status in ("stopped", "disconnected"):
                            return None if pan_status == "stopped" else 0.0
                    else:
                        self._emit_log(
                            "WARN",
                            f"[pan-{direction}] Tree coverage dropped to {pan_scan.tree_coverage_ratio:.2f} without Start anchor; aborting pan.",
                        )

                # Pan back and visually recenter onto Start when visible
                self.controller.drag_client_points(x2, y2, x1, y1, hwnd=hwnd, dry_run=False)
                self.controller.park_cursor_neutral(hwnd=hwnd, dry_run=False)
                if zoom_settle > 0 and self.stop_event.wait(zoom_settle):
                    return None

                st_back, hwnd, back_frame, back_scan = _capture_and_scan()
                if st_back in ("stopped", "disconnected"):
                    return None if st_back == "stopped" else 0.0
                if st_back == "ok" and back_frame is not None and back_scan is not None:
                    if cfg.adaptive_zoom and back_scan.start_anchor is not None:
                        st_rec, hwnd, back_frame, back_scan = _recenter_on_start(hwnd, back_frame, back_scan)
                        if st_rec in ("stopped", "disconnected"):
                            return None if st_rec == "stopped" else 0.0
                    cur_frame, cur_scan = back_frame, back_scan

                if pan_status in ("tree_closed", "budget_exhausted"):
                    break

        # 5. Close upgrade tree (up to 3 attempts, Esc fallback)
        close_status = self._close_modal_via_template(
            close_template="btn_upgrades_close",
            close_threshold=th.btn_upgrades_close,
            close_roi=rois.btn_upgrades_close,
            disconnect_reason="modal_disconnected_upgrading",
        )
        if close_status == "stopped":
            return None
        if close_status == "disconnected":
            return 0.0

        self._emit_log(
            "INFO",
            f"Auto-Upgrade pass complete: bought {purchases_in_pass} upgrade(s) "
            f"(session total: {self.stats.upgrades_purchased}).",
        )
        self._transition_to(MacroState.READY_TO_START)
        return 0.0

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
            self._record_run_completed("Battle button reappeared")
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
                    self._run_end_unconfirmed = True
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

    def _record_run_completed(self, reason: str) -> None:
        """Count a finished run and queue the cadence-based Spawner/Upgrade passes."""
        self._run_end_unconfirmed = False
        self.stats.runs_completed += 1
        self.stats.consecutive_recoveries = 0
        self._recovery_timestamps.clear()
        self._emit_log("INFO", f"Run #{self.stats.runs_completed} completed ({reason}).")
        spawner_every_n = max(1, int(self.config.auto_spawner.every_n_runs))
        if self._can_run_auto_spawners() and (self.stats.runs_completed % spawner_every_n == 0):
            self._pending_spawner_pass = True
        upgrade_every_n = max(1, int(self.config.auto_upgrade.every_n_runs))
        if self._can_run_auto_upgrade() and (self.stats.runs_completed % upgrade_every_n == 0):
            self._pending_upgrade_pass = True
        self._emit_metrics()

    def _step_disconnected(self) -> Optional[float]:
        menu_poll = self.config.polling.menu_interval_sec
        hwnd, frame = self._capture_current_frame()

        # Feature 1: Same-account disconnect guard (Error 273) before reconnect/relaunch
        if (
            frame is not None
            and self.config.recovery.stop_on_same_account
            and self.vision.has_template("modal_same_account")
        ):
            same_acct = self.vision.find(
                frame,
                "modal_same_account",
                self.config.thresholds.modal_same_account,
                self.config.rois.modal_same_account,
            )
            if same_acct.found:
                if not self._disconnect_handled:
                    self._disconnect_handled = True
                    self.stats.disconnects += 1
                snap = self.vision.save_failure_snapshot(frame, "same_account_disconnect")
                snap_msg = f" Saved snapshot {snap.name}." if snap is not None else ""
                self._stop_reason = "Same account launched on another device (Error 273)"
                self._emit_log(
                    "WARN",
                    "Detected Roblox Error 273 (same account launched on another device). "
                    f"Stopping macro cleanly without reconnecting or relaunching Roblox.{snap_msg}",
                )
                self._emit_metrics()
                self._transition_to(MacroState.IDLE)
                self.stop_event.set()
                return None

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
