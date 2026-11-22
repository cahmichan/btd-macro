"""Comprehensive unit tests for every MacroStateMachine transition, watchdog, and circuit breaker."""

from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.config import AppConfig
from src.state_machine import MacroState, MacroStateMachine
from src.vision import MatchResult


class FakeClock:
    """Deterministic monotonic clock for FSM tests."""

    def __init__(self, initial: float = 1000.0) -> None:
        self.now = initial

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeVision:
    """Controllable mock of VisionEngine."""

    def __init__(self, missing: Optional[list[str]] = None) -> None:
        self.missing = list(missing or [])
        self.loaded_optional: set[str] = set()
        self.frame_seq: list[Optional[np.ndarray]] = []
        self.matches: dict[str, MatchResult] = {}
        self.match_seq: dict[str, list[MatchResult]] = {}
        self.dismiss_result: Optional[MatchResult] = None
        self.saved_snapshots: list[str] = []

    def reload_templates(self) -> None:
        pass

    def missing_required_templates(self) -> list[str]:
        return list(self.missing)

    def has_template(self, name: str) -> bool:
        if name == "hud_in_run":
            return (
                name in self.loaded_optional
                or name in self.matches
                or name in self.match_seq
            )
        return name not in self.missing

    def capture_client(self, client_rect: tuple[int, int, int, int]) -> Optional[np.ndarray]:
        if self.frame_seq:
            return self.frame_seq.pop(0)
        return np.zeros((720, 1280, 3), dtype=np.uint8)

    def find(
        self,
        frame: Optional[np.ndarray],
        name: str,
        threshold: float = 0.85,
        roi: Any = None,
    ) -> MatchResult:
        seq = self.match_seq.get(name)
        if seq:
            return seq.pop(0)
        return self.matches.get(name, MatchResult(found=False, confidence=0.1, template_name=name))

    def find_dismiss_target(self, frame: Optional[np.ndarray], threshold: float = 0.85) -> Optional[MatchResult]:
        return self.dismiss_result

    def save_failure_snapshot(self, frame: Optional[np.ndarray], reason: str) -> Optional[Path]:
        self.saved_snapshots.append(reason)
        return Path(f"logs/failures/fake_{reason}.png")


class FakeController:
    """Controllable mock of RobloxController."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.roblox_running = True
        self.launcher_running = False
        self.hwnd: Optional[int] = 12345
        self.client_rect: Optional[tuple[int, int, int, int]] = (0, 30, 1280, 690)
        self.launch_calls = 0
        self.terminate_calls = 0
        self.layout_calls = 0
        self.focus_calls = 0
        self.clicks: list[tuple[int, int, bool]] = []
        self.anti_afk_calls: list[bool] = []
        self.click_returns = True

    def is_roblox_running(self) -> bool:
        return self.roblox_running

    def is_launcher_running(self) -> bool:
        return self.launcher_running

    def get_hwnd(self) -> Optional[int]:
        return self.hwnd

    def get_client_rect(self, hwnd: Optional[int] = None) -> Optional[tuple[int, int, int, int]]:
        return self.client_rect

    def ensure_window_layout(self, hwnd: Optional[int] = None) -> bool:
        self.layout_calls += 1
        return True

    def focus(self, hwnd: Optional[int] = None) -> bool:
        self.focus_calls += 1
        return True

    def launch_game(self) -> bool:
        self.launch_calls += 1
        return True

    def terminate_roblox(self, timeout_sec: float = 5.0) -> int:
        self.terminate_calls += 1
        self.roblox_running = False
        self.hwnd = None
        return 1

    def click_client_point(
        self,
        client_x: int,
        client_y: int,
        hwnd: Optional[int] = None,
        dry_run: bool = False,
    ) -> bool:
        self.clicks.append((client_x, client_y, dry_run))
        return self.click_returns

    def send_anti_afk(self, hwnd: Optional[int] = None, dry_run: bool = False) -> bool:
        self.anti_afk_calls.append(dry_run)
        return True


def _build_fsm(
    clock: Optional[FakeClock] = None,
    missing: Optional[list[str]] = None,
) -> tuple[MacroStateMachine, FakeVision, FakeController, FakeClock, list[str]]:
    cfg = AppConfig()
    cfg.polling.post_launch_settle_sec = 0.0
    cfg.polling.click_confirm_poll_sec = 0.001
    cfg.polling.click_confirm_timeout_sec = 0.005

    clk = clock or FakeClock()
    vision = FakeVision(missing=missing)
    controller = FakeController(cfg)
    sleep_calls: list[str] = []
    q: queue.Queue[dict[str, Any]] = queue.Queue()

    fsm = MacroStateMachine(
        config=cfg,
        vision=vision,  # type: ignore[arg-type]
        controller=controller,  # type: ignore[arg-type]
        event_queue=q,
        clock=clk,
        sleep_prevent_fn=lambda: sleep_calls.append("prevent") or True,
        sleep_allow_fn=lambda: sleep_calls.append("allow") or True,
    )
    return fsm, vision, controller, clk, sleep_calls


def test_start_refused_if_battle_template_missing() -> None:
    fsm, _, _, _, _ = _build_fsm(missing=["btn_battle"])
    started = fsm.start(dry_run=False)
    assert started is False
    assert fsm.state == MacroState.IDLE


def test_happy_path_full_run_loop_and_anti_afk() -> None:
    fsm, vision, ctrl, clk, _ = _build_fsm()
    fsm._transition_to(MacroState.LAUNCHING)

    # LAUNCHING -> READY_TO_START
    delay = fsm.step()
    assert delay == 0.0
    assert fsm.state == MacroState.READY_TO_START
    assert ctrl.layout_calls == 1
    assert ctrl.focus_calls == 1

    # READY_TO_START sees btn_battle, clicks it, then confirm poll sees it disappear -> IN_RUN
    battle_hit = MatchResult(found=True, confidence=0.97, center=(680, 620), template_name="btn_battle")
    battle_miss = MatchResult(found=False, confidence=0.12, template_name="btn_battle")
    vision.match_seq["btn_battle"] = [battle_hit, battle_miss]

    delay = fsm.step()
    assert fsm.state == MacroState.IN_RUN
    assert delay == fsm.config.polling.in_run_interval_sec
    assert ctrl.clicks == [(680, 620, False)]

    # During IN_RUN before anti-AFK interval: no anti-AFK yet
    vision.matches["btn_battle"] = battle_miss
    clk.advance(60.0)
    fsm.step()
    assert ctrl.anti_afk_calls == []

    # After 720s: anti-AFK fires
    clk.advance(720.0)
    fsm.step()
    assert ctrl.anti_afk_calls == [False]

    # When btn_battle reappears in IN_RUN: run completes -> READY_TO_START
    vision.matches["btn_battle"] = battle_hit
    delay = fsm.step()
    assert delay == 0.0
    assert fsm.state == MacroState.READY_TO_START
    assert fsm.stats.runs_completed == 1


def test_missed_click_retries_then_triggers_watchdog() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    fsm.config.recovery.max_click_retries = 2
    fsm._transition_to(MacroState.READY_TO_START)

    battle_hit = MatchResult(found=True, confidence=0.95, center=(680, 620), template_name="btn_battle")
    vision.matches["btn_battle"] = battle_hit

    # Attempt 1: click fails confirmation
    fsm.step()
    assert fsm.state == MacroState.READY_TO_START
    assert len(ctrl.clicks) == 1

    # Attempt 2: reaches max_click_retries -> WATCHDOG_RECOVERY
    delay = fsm.step()
    assert delay == 0.0
    assert fsm.state == MacroState.WATCHDOG_RECOVERY


def test_launching_update_grace_period_with_launcher_process() -> None:
    fsm, _, ctrl, clk, _ = _build_fsm()
    ctrl.roblox_running = False
    ctrl.hwnd = None
    ctrl.launcher_running = True
    fsm._transition_to(MacroState.LAUNCHING)

    # First step detects RobloxPlayerLauncher.exe and activates grace
    fsm.step()
    assert fsm._update_grace_active is True
    assert ctrl.launch_calls == 0

    # Past normal launching_sec (180s), still inside 180+300=480s grace -> stays in LAUNCHING
    clk.advance(250.0)
    fsm.step()
    assert fsm.state == MacroState.LAUNCHING

    # Past 480s total -> triggers WATCHDOG_RECOVERY
    clk.advance(250.0)
    fsm.step()
    assert fsm.state == MacroState.WATCHDOG_RECOVERY


def test_disconnect_reconnect_success_and_retry_exhaustion() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    fsm._transition_to(MacroState.IN_RUN)

    disc_hit = MatchResult(found=True, confidence=0.95, center=(640, 260), template_name="modal_disconnected")
    disc_miss = MatchResult(found=False, confidence=0.10, template_name="modal_disconnected")
    recon_hit = MatchResult(found=True, confidence=0.93, center=(700, 440), template_name="btn_reconnect")
    battle_hit = MatchResult(found=True, confidence=0.95, center=(680, 620), template_name="btn_battle")

    vision.matches["modal_disconnected"] = disc_hit
    fsm.step()
    assert fsm.state == MacroState.DISCONNECTED

    # Attempt 1 clicks Reconnect
    vision.matches["btn_reconnect"] = recon_hit
    delay = fsm.step()
    assert delay == fsm.config.recovery.reconnect_wait_sec
    assert fsm.stats.disconnects == 1
    assert vision.saved_snapshots == ["modal_disconnected_in_run"]
    assert ctrl.clicks[-1] == (700, 440, False)

    # Modal clears and Battle returns -> recovers to READY_TO_START
    vision.matches["modal_disconnected"] = disc_miss
    vision.matches["btn_battle"] = battle_hit
    delay = fsm.step()
    assert delay == 0.0
    assert fsm.state == MacroState.READY_TO_START
    assert ctrl.terminate_calls == 0

    # Now test retry exhaustion (max_reconnect_retries=2)
    fsm._transition_to(MacroState.DISCONNECTED)
    vision.matches["modal_disconnected"] = disc_hit
    vision.matches["btn_battle"] = MatchResult(found=False, confidence=0.1, template_name="btn_battle")
    vision.matches["btn_reconnect"] = recon_hit
    fsm.step()  # retry 1
    fsm.step()  # retry 2
    assert fsm.state == MacroState.DISCONNECTED
    delay = fsm.step()  # exhausted -> terminate & LAUNCHING
    assert delay == 0.0
    assert fsm.state == MacroState.LAUNCHING
    assert ctrl.terminate_calls == 1


def test_disconnect_leave_only_modal_and_process_crash() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    fsm._transition_to(MacroState.DISCONNECTED)

    vision.matches["modal_disconnected"] = MatchResult(found=True, confidence=0.95, template_name="modal_disconnected")
    vision.matches["btn_reconnect"] = MatchResult(found=False, confidence=0.10, template_name="btn_reconnect")

    delay = fsm.step()
    assert delay == 0.0
    assert fsm.state == MacroState.LAUNCHING
    assert ctrl.terminate_calls == 1

    # Process crash during IN_RUN -> DISCONNECTED -> cleans up to LAUNCHING
    ctrl.roblox_running = False
    ctrl.hwnd = None
    fsm._transition_to(MacroState.IN_RUN)
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.DISCONNECTED
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.LAUNCHING
    assert "process_crashed_in_run" in vision.saved_snapshots


def test_per_state_watchdogs_and_circuit_breaker() -> None:
    fsm, vision, ctrl, clk, _ = _build_fsm()

    # IN_RUN watchdog timeout (2700s)
    fsm._transition_to(MacroState.IN_RUN)
    clk.advance(2705.0)
    fsm.step()
    assert fsm.state == MacroState.WATCHDOG_RECOVERY

    # Trigger 5 consecutive recoveries within 900s -> circuit breaker trips on 5th
    for i in range(4):
        assert fsm.state == MacroState.WATCHDOG_RECOVERY
        delay = fsm.step()
        assert delay == 0.0
        assert fsm.state == MacroState.LAUNCHING
        assert fsm.stats.circuit_breaker_tripped is False
        clk.advance(10.0)
        fsm._transition_to(MacroState.WATCHDOG_RECOVERY)

    # 5th recovery trips breaker and halts cleanly
    delay = fsm.step()
    assert delay is None
    assert fsm.state == MacroState.IDLE
    assert fsm.stats.circuit_breaker_tripped is True
    assert fsm.stop_event.is_set()
    assert len(vision.saved_snapshots) == 5
    assert ctrl.terminate_calls == 5


def test_circuit_breaker_resets_after_run_completion_or_window_expiry() -> None:
    fsm, vision, ctrl, clk, _ = _build_fsm()

    # 4 recoveries, then 1 successful run resets streak
    for _ in range(4):
        fsm._transition_to(MacroState.WATCHDOG_RECOVERY)
        fsm.step()
        clk.advance(5.0)
    assert fsm.stats.consecutive_recoveries == 4

    ctrl.roblox_running = True
    ctrl.hwnd = 12345
    fsm._transition_to(MacroState.IN_RUN)
    vision.matches["btn_battle"] = MatchResult(found=True, confidence=0.96, center=(680, 620), template_name="btn_battle")
    fsm.step()
    assert fsm.stats.runs_completed == 1
    assert fsm.stats.consecutive_recoveries == 0

    # Recoveries spaced >900s apart prune old timestamps and never trip
    for _ in range(6):
        fsm._transition_to(MacroState.WATCHDOG_RECOVERY)
        delay = fsm.step()
        assert delay == 0.0
        assert fsm.stats.circuit_breaker_tripped is False
        clk.advance(950.0)


def test_dry_run_never_clicks_launches_or_kills() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    fsm.dry_run = True
    fsm._transition_to(MacroState.LAUNCHING)
    fsm.step()
    assert fsm.state == MacroState.READY_TO_START

    vision.matches["btn_battle"] = MatchResult(found=True, confidence=0.95, center=(680, 620), template_name="btn_battle")
    fsm.step()
    assert fsm.state == MacroState.READY_TO_START
    assert ctrl.clicks == []
    assert ctrl.launch_calls == 0
    assert ctrl.terminate_calls == 0


def test_background_thread_instant_stop_and_sleep_lock_release() -> None:
    fsm, vision, _, _, sleep_calls = _build_fsm()
    fsm.config.polling.in_run_interval_sec = 30.0
    fsm.config.polling.menu_interval_sec = 0.05

    battle_hit = MatchResult(found=True, confidence=0.96, center=(680, 620), template_name="btn_battle")
    battle_miss = MatchResult(found=False, confidence=0.10, template_name="btn_battle")
    vision.match_seq["btn_battle"] = [battle_hit, battle_miss]

    assert fsm.start(dry_run=False) is True

    # Wait briefly until FSM enters IN_RUN (waiting 30s on stop_event.wait)
    deadline = time.monotonic() + 2.0
    while fsm.state != MacroState.IN_RUN and time.monotonic() < deadline:
        time.sleep(0.01)
    assert fsm.state == MacroState.IN_RUN

    t0 = time.monotonic()
    fsm.stop("Instant stop test")
    elapsed = time.monotonic() - t0

    assert elapsed < 0.5
    assert not fsm.is_running
    assert fsm.state == MacroState.IDLE
    assert sleep_calls == ["prevent", "allow"]


def test_mid_run_disconnect_resumes_with_yes_continue_by_default() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    assert fsm.config.recovery.resume_mode == "continue"
    fsm._transition_to(MacroState.DISCONNECTED)

    disc_miss = MatchResult(found=False, confidence=0.10, template_name="modal_disconnected")
    yes_hit = MatchResult(found=True, confidence=0.96, center=(580, 310), template_name="btn_yes_continue")
    yes_miss = MatchResult(found=False, confidence=0.10, template_name="btn_yes_continue")
    battle_hit = MatchResult(found=True, confidence=0.96, center=(680, 620), template_name="btn_battle")

    vision.matches["modal_disconnected"] = disc_miss
    vision.matches["btn_yes_continue"] = yes_hit

    # DISCONNECTED sees Yes, continue -> transitions to READY_TO_START
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.READY_TO_START

    # READY_TO_START clicks Yes, continue and confirms it disappears -> enters IN_RUN without incrementing runs_completed
    vision.match_seq["btn_yes_continue"] = [yes_hit, yes_miss]
    fsm.step()
    assert fsm.state == MacroState.IN_RUN
    assert fsm.stats.runs_completed == 0
    assert ctrl.clicks == [(580, 310, False)]

    # Later when battle actually finishes and btn_battle reappears -> runs_completed increments to 1
    vision.matches["btn_yes_continue"] = yes_miss
    vision.matches["btn_battle"] = battle_hit
    fsm.step()
    assert fsm.state == MacroState.READY_TO_START
    assert fsm.stats.runs_completed == 1


def test_mid_run_bottom_continue_button_opens_modal_then_resumes() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    fsm._transition_to(MacroState.READY_TO_START)

    yes_hit = MatchResult(found=True, confidence=0.95, center=(580, 310), template_name="btn_yes_continue")
    yes_miss = MatchResult(found=False, confidence=0.10, template_name="btn_yes_continue")
    cont_hit = MatchResult(found=True, confidence=0.94, center=(580, 500), template_name="btn_continue")
    cont_miss = MatchResult(found=False, confidence=0.10, template_name="btn_continue")

    # Step 1: only bottom Continue button is visible -> click it to open confirmation modal
    vision.matches["btn_yes_continue"] = yes_miss
    vision.matches["btn_continue"] = cont_hit
    fsm.step()
    assert fsm.state == MacroState.READY_TO_START
    assert ctrl.clicks == [(580, 500, False)]

    # Step 2: confirmation modal is now open -> click Yes, continue and enter IN_RUN
    vision.matches["btn_continue"] = cont_miss
    vision.match_seq["btn_yes_continue"] = [yes_hit, yes_miss]
    fsm.step()
    assert fsm.state == MacroState.IN_RUN
    assert fsm.stats.runs_completed == 0
    assert ctrl.clicks == [(580, 500, False), (580, 310, False)]


def test_late_resume_modal_in_run_does_not_count_as_completed_run() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    fsm._transition_to(MacroState.IN_RUN)

    yes_hit = MatchResult(found=True, confidence=0.95, center=(580, 310), template_name="btn_yes_continue")
    yes_miss = MatchResult(found=False, confidence=0.10, template_name="btn_yes_continue")
    vision.match_seq["btn_yes_continue"] = [yes_hit, yes_hit, yes_miss]

    # IN_RUN detects late resume modal -> transitions to READY_TO_START without incrementing runs_completed
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.READY_TO_START
    assert fsm.stats.runs_completed == 0

    # READY_TO_START clicks Yes, continue -> back to IN_RUN
    fsm.step()
    assert fsm.state == MacroState.IN_RUN
    assert ctrl.clicks == [(580, 310, False)]


def test_mid_run_start_over_mode_clicks_start_over() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    fsm.config.recovery.resume_mode = "start_over"
    fsm._transition_to(MacroState.READY_TO_START)

    start_over_hit = MatchResult(found=True, confidence=0.95, center=(440, 310), template_name="btn_start_over")
    start_over_miss = MatchResult(found=False, confidence=0.10, template_name="btn_start_over")
    vision.match_seq["btn_start_over"] = [start_over_hit, start_over_miss]

    fsm.step()
    assert fsm.state == MacroState.IN_RUN
    assert ctrl.clicks == [(440, 310, False)]


def test_starting_mid_run_detects_hud_in_run_without_clicking() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    fsm._transition_to(MacroState.LAUNCHING)
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.READY_TO_START

    vision.matches["hud_in_run"] = MatchResult(found=True, confidence=0.96, center=(820, 620), template_name="hud_in_run")
    delay = fsm.step()
    assert fsm.state == MacroState.IN_RUN
    assert delay == fsm.config.polling.in_run_interval_sec
    assert ctrl.clicks == []
    assert fsm.stats.runs_completed == 0


def test_click_confirmation_waits_for_hud_in_run_when_template_loaded() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    fsm._transition_to(MacroState.READY_TO_START)

    battle_hit = MatchResult(found=True, confidence=0.96, center=(680, 620), template_name="btn_battle")
    battle_miss = MatchResult(found=False, confidence=0.10, template_name="btn_battle")
    hud_miss = MatchResult(found=False, confidence=0.10, template_name="hud_in_run")
    hud_hit = MatchResult(found=True, confidence=0.96, center=(820, 620), template_name="hud_in_run")

    # Battle disappears on poll 1 while HUD is still transitioning, then HUD appears on poll 2
    vision.match_seq["btn_battle"] = [battle_hit, battle_miss, battle_miss]
    vision.match_seq["hud_in_run"] = [hud_miss, hud_miss, hud_hit]

    delay = fsm.step()
    assert fsm.state == MacroState.IN_RUN
    assert delay == fsm.config.polling.in_run_interval_sec
    assert ctrl.clicks == [(680, 620, False)]


def test_battle_click_opening_continue_prompt_returns_to_ready_to_start() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    fsm._transition_to(MacroState.READY_TO_START)

    battle_hit = MatchResult(found=True, confidence=0.95, center=(680, 620), template_name="btn_battle")
    battle_miss = MatchResult(found=False, confidence=0.10, template_name="btn_battle")
    yes_miss = MatchResult(found=False, confidence=0.10, template_name="btn_yes_continue")
    yes_hit = MatchResult(found=True, confidence=0.95, center=(580, 310), template_name="btn_yes_continue")

    vision.match_seq["btn_battle"] = [battle_hit, battle_miss]
    vision.match_seq["btn_yes_continue"] = [yes_miss, yes_hit]

    delay = fsm.step()
    assert delay == 0.0
    assert fsm.state == MacroState.READY_TO_START
    assert ctrl.clicks == [(680, 620, False)]


def test_disconnect_recovers_straight_to_in_run_when_hud_in_run_visible() -> None:
    fsm, vision, _, _, _ = _build_fsm()
    fsm._transition_to(MacroState.DISCONNECTED)

    vision.matches["modal_disconnected"] = MatchResult(found=False, confidence=0.10, template_name="modal_disconnected")
    vision.matches["hud_in_run"] = MatchResult(found=True, confidence=0.96, center=(820, 620), template_name="hud_in_run")

    assert fsm.step() == 0.0
    assert fsm.state == MacroState.IN_RUN


def test_dry_run_tracks_state_transitions_between_ready_and_in_run() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    fsm.dry_run = True
    fsm._transition_to(MacroState.READY_TO_START)

    hud_hit = MatchResult(found=True, confidence=0.96, center=(820, 620), template_name="hud_in_run")
    hud_miss = MatchResult(found=False, confidence=0.10, template_name="hud_in_run")
    battle_hit = MatchResult(found=True, confidence=0.96, center=(680, 620), template_name="btn_battle")
    battle_miss = MatchResult(found=False, confidence=0.10, template_name="btn_battle")

    # Wave active -> switches to IN_RUN
    vision.matches["hud_in_run"] = hud_hit
    vision.matches["btn_battle"] = battle_miss
    fsm.step()
    assert fsm.state == MacroState.IN_RUN

    # Wave ends & Battle returns -> switches back to READY_TO_START without clicking or incrementing runs_completed
    vision.matches["hud_in_run"] = hud_miss
    vision.matches["btn_battle"] = battle_hit
    fsm.step()
    assert fsm.state == MacroState.READY_TO_START
    assert ctrl.clicks == []
    assert fsm.stats.runs_completed == 0


def test_in_run_recovers_to_ready_when_hud_disappears_without_battle_match() -> None:
    fsm, vision, _, _, _ = _build_fsm()
    fsm._transition_to(MacroState.IN_RUN)

    vision.matches["hud_in_run"] = MatchResult(found=False, confidence=0.10, template_name="hud_in_run")
    vision.matches["btn_battle"] = MatchResult(found=False, confidence=0.50, template_name="btn_battle")

    # Polls 1 and 2 stay in IN_RUN
    fsm.step()
    assert fsm.state == MacroState.IN_RUN
    fsm.step()
    assert fsm.state == MacroState.IN_RUN

    # Poll 3 (streak == 3) returns to READY_TO_START
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.READY_TO_START
