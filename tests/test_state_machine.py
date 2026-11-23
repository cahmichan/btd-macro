"""Comprehensive unit tests for every MacroStateMachine transition, watchdog, and circuit breaker."""

from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.config import AppConfig
from src.spawner_vision import SpawnerBuyButton, SpawnerScanResult
from src.state_machine import MacroState, MacroStateMachine
from src.upgrade_vision import StartAnchor, TreeScanResult, UpgradeCandidate
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

    OPTIONAL_CHECK_NAMES = {
        "hud_in_run",
        "modal_same_account",
        "btn_upgrades",
        "btn_upgrades_close",
        "btn_spawners",
        "btn_spawners_close",
    }

    def has_template(self, name: str) -> bool:
        if name in self.OPTIONAL_CHECK_NAMES:
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
        self.hovers: list[tuple[int, int, bool]] = []
        self.scrolls: list[tuple[int, int, int, bool]] = []
        self.drags: list[tuple[int, int, int, int, bool]] = []
        self.parks: list[bool] = []
        self.keys: list[tuple[str, bool]] = []
        self.anti_afk_calls: list[bool] = []
        self.click_returns = True
        self.hover_returns = True

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

    def hover_client_point(
        self,
        client_x: int,
        client_y: int,
        hwnd: Optional[int] = None,
        dry_run: bool = False,
    ) -> bool:
        self.hovers.append((client_x, client_y, dry_run))
        return self.hover_returns

    def click_client_point(
        self,
        client_x: int,
        client_y: int,
        hwnd: Optional[int] = None,
        dry_run: bool = False,
        park_cursor: bool = True,
    ) -> bool:
        self.clicks.append((client_x, client_y, dry_run))
        return self.click_returns

    def scroll_client_point(
        self,
        client_x: int,
        client_y: int,
        notches: int,
        hwnd: Optional[int] = None,
        dry_run: bool = False,
    ) -> bool:
        self.scrolls.append((client_x, client_y, notches, dry_run))
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
        self.drags.append((x1, y1, x2, y2, dry_run))
        return True

    def park_cursor_neutral(self, hwnd: Optional[int] = None, dry_run: bool = False) -> bool:
        self.parks.append(dry_run)
        return True

    def send_key(self, key: str, hwnd: Optional[int] = None, dry_run: bool = False) -> bool:
        self.keys.append((key, dry_run))
        return True

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


def _make_candidate(x: int, y: int, is_main: bool = True, dist: float = 20.0) -> UpgradeCandidate:
    return UpgradeCandidate(
        price_center=(x, y + 14),
        click_point=(x, y),
        bbox=(x - 10, y + 10, 20, 6),
        is_main_tree=is_main,
        distance_from_center=dist,
        raw_pixels=36,
    )


def _enable_auto_upgrade(
    fsm: MacroStateMachine,
    vision: FakeVision,
    scope: str = "main_only",
    every_n_runs: int = 1,
    run_on_start: bool = True,
    adaptive_zoom: bool = False,
) -> None:
    fsm.config.auto_upgrade.enabled = True
    fsm.config.auto_upgrade.scope = scope
    fsm.config.auto_upgrade.every_n_runs = every_n_runs
    fsm.config.auto_upgrade.run_on_start = run_on_start
    fsm.config.auto_upgrade.post_click_settle_sec = 0.0
    fsm.config.auto_upgrade.hover_settle_sec = 0.0
    fsm.config.auto_upgrade.adaptive_zoom = adaptive_zoom
    vision.loaded_optional.update({"btn_upgrades", "btn_upgrades_close"})
    # Default safe mock OCR & tooltip cropper so upgrade click tests focus on FSM logic
    if fsm._ocr_engine is None:
        fsm._ocr_engine = lambda crop, **kwargs: ["Damage XII", "+25% Damage", "$50.0M"]
    if fsm.tooltip_extractor.__name__ == "extract_hover_tooltip_crop":
        fsm.tooltip_extractor = lambda before, after, pt: np.full((60, 180, 3), 35, dtype=np.uint8)


def _enable_auto_spawners(
    fsm: MacroStateMachine,
    vision: FakeVision,
    every_n_runs: int = 1,
    run_on_start: bool = True,
    verify_buy_text_ocr: bool = False,
) -> None:
    fsm.config.auto_spawner.enabled = True
    fsm.config.auto_spawner.every_n_runs = every_n_runs
    fsm.config.auto_spawner.run_on_start = run_on_start
    fsm.config.auto_spawner.click_hover_sec = 0.0
    fsm.config.auto_spawner.click_interval_sec = 0.0
    fsm.config.auto_spawner.round_settle_sec = 0.0
    vision.loaded_optional.update({"btn_spawners", "btn_spawners_close"})


def test_same_account_disconnect_error_273_stops_cleanly_without_relaunch() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    fsm.config.recovery.stop_on_same_account = True
    vision.loaded_optional.add("modal_same_account")
    vision.matches["modal_disconnected"] = MatchResult(
        found=True, confidence=0.95, center=(640, 260), template_name="modal_disconnected"
    )
    vision.matches["modal_same_account"] = MatchResult(
        found=True, confidence=0.94, center=(640, 320), template_name="modal_same_account"
    )
    vision.matches["btn_reconnect"] = MatchResult(
        found=True, confidence=0.93, center=(700, 440), template_name="btn_reconnect"
    )

    fsm._transition_to(MacroState.IN_RUN)
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.DISCONNECTED

    delay = fsm.step()
    assert delay is None
    assert fsm.state == MacroState.IDLE
    assert fsm.stop_event.is_set()
    assert fsm.stats.disconnects == 1
    assert ctrl.clicks == []
    assert ctrl.terminate_calls == 0
    assert ctrl.launch_calls == 0
    assert vision.saved_snapshots == ["same_account_disconnect"]

    # _run_loop emits stopped reason for Error 273
    fsm.stop_event.clear()
    fsm._transition_to(MacroState.DISCONNECTED)
    fsm._run_loop()
    events = []
    assert fsm.event_queue is not None
    while not fsm.event_queue.empty():
        events.append(fsm.event_queue.get_nowait())
    stopped_events = [e for e in events if e.get("type") == "stopped"]
    assert stopped_events
    assert "Error 273" in stopped_events[-1]["reason"]


def test_same_account_disconnect_guard_can_be_disabled() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    fsm.config.recovery.stop_on_same_account = False
    vision.loaded_optional.add("modal_same_account")
    vision.matches["modal_disconnected"] = MatchResult(
        found=True, confidence=0.95, center=(640, 260), template_name="modal_disconnected"
    )
    vision.matches["modal_same_account"] = MatchResult(
        found=True, confidence=0.94, center=(640, 320), template_name="modal_same_account"
    )
    vision.matches["btn_reconnect"] = MatchResult(
        found=True, confidence=0.93, center=(700, 440), template_name="btn_reconnect"
    )

    fsm._transition_to(MacroState.DISCONNECTED)
    delay = fsm.step()
    assert delay == fsm.config.recovery.reconnect_wait_sec
    assert fsm.state == MacroState.DISCONNECTED
    assert not fsm.stop_event.is_set()
    assert ctrl.clicks == [(700, 440, False)]


def test_auto_upgrade_startup_and_every_n_runs_cadence() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    _enable_auto_upgrade(fsm, vision, every_n_runs=2, run_on_start=True)

    scan_empty = TreeScanResult(candidates=[], all_candidates=[], main_tree_bbox=None, clipped_edges=set())
    fsm.upgrade_scanner = lambda frame, **kwargs: scan_empty  # type: ignore[assignment]

    battle_hit = MatchResult(found=True, confidence=0.96, center=(680, 620), template_name="btn_battle")
    battle_miss = MatchResult(found=False, confidence=0.10, template_name="btn_battle")
    up_hit = MatchResult(found=True, confidence=0.95, center=(200, 620), template_name="btn_upgrades")
    close_hit = MatchResult(found=True, confidence=0.95, center=(640, 660), template_name="btn_upgrades_close")
    close_miss = MatchResult(found=False, confidence=0.10, template_name="btn_upgrades_close")

    # Startup triggers UPGRADING once battle button confirms lobby
    fsm._pending_upgrade_pass = True
    fsm._transition_to(MacroState.READY_TO_START)
    vision.matches["btn_battle"] = battle_hit
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.UPGRADING

    # Execute UPGRADING pass (open -> scan empty -> close -> READY_TO_START)
    vision.matches["btn_upgrades"] = up_hit
    vision.match_seq["btn_upgrades_close"] = [close_miss, close_hit, close_hit, close_hit, close_miss]
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.READY_TO_START

    # Run #1 completes (every_n_runs=2 -> should NOT trigger upgrade pass)
    fsm._transition_to(MacroState.IN_RUN)
    vision.matches["btn_battle"] = battle_hit
    assert fsm.step() == 0.0
    assert fsm.stats.runs_completed == 1
    assert fsm._pending_upgrade_pass is False

    # Run #2 completes (every_n_runs=2 -> triggers upgrade pass when entering READY_TO_START)
    fsm._transition_to(MacroState.IN_RUN)
    assert fsm.step() == 0.0
    assert fsm.stats.runs_completed == 2
    assert fsm._pending_upgrade_pass is True
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.UPGRADING


def test_auto_upgrade_happy_path_buys_main_and_newly_revealed_tiles() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    _enable_auto_upgrade(fsm, vision, scope="main_only")

    c1 = _make_candidate(540, 300, is_main=True, dist=15.0)
    c2_revealed = _make_candidate(580, 260, is_main=True, dist=30.0)
    scans = [
        TreeScanResult([c1], [c1], (400, 150, 480, 420), set()),
        TreeScanResult([c2_revealed], [c2_revealed], (400, 150, 480, 420), set()),
        TreeScanResult([], [], (400, 150, 480, 420), set()),
    ]
    fsm.upgrade_scanner = lambda frame, **kwargs: scans.pop(0) if scans else TreeScanResult([], [], (400, 150, 480, 420), set())  # type: ignore[assignment]

    up_hit = MatchResult(found=True, confidence=0.95, center=(200, 620), template_name="btn_upgrades")
    close_hit = MatchResult(found=True, confidence=0.95, center=(640, 660), template_name="btn_upgrades_close")
    close_miss = MatchResult(found=False, confidence=0.10, template_name="btn_upgrades_close")
    battle_hit = MatchResult(found=True, confidence=0.96, center=(680, 620), template_name="btn_battle")
    battle_miss = MatchResult(found=False, confidence=0.10, template_name="btn_battle")

    vision.matches["btn_upgrades"] = up_hit
    # Open check (miss -> hit), post-zoom scan (hit), post-click c1 (hit), post-click c2 (hit), close check (hit), confirm close (miss)
    vision.match_seq["btn_upgrades_close"] = [close_miss, close_hit, close_hit, close_hit, close_hit, close_hit, close_miss]

    fsm._transition_to(MacroState.UPGRADING)
    delay = fsm.step()
    assert delay == 0.0
    assert fsm.state == MacroState.READY_TO_START
    assert fsm.stats.upgrades_purchased == 2
    assert ctrl.scrolls == [(640, 345, -15, False), (640, 345, 5, False)]
    # first hover is the Upgrades button (open step hovers before clicking), then each tile
    assert ctrl.hovers == [(200, 620, False), (540, 300, False), (580, 260, False)]
    assert ctrl.clicks == [(200, 620, False), (540, 300, False), (580, 260, False), (640, 660, False)]
    assert len(ctrl.parks) >= 3

    # Next READY_TO_START tick clicks Battle normally
    vision.match_seq["btn_battle"] = [battle_hit, battle_miss]
    fsm.step()
    assert fsm.state == MacroState.IN_RUN


def test_auto_upgrade_main_then_sides_and_edge_panning() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    _enable_auto_upgrade(fsm, vision, scope="main_then_sides")
    fsm.config.auto_upgrade.enable_edge_panning = True
    fsm.config.auto_upgrade.pan_drag_px = 200

    c_main = _make_candidate(640, 340, is_main=True, dist=10.0)
    c_side = _make_candidate(1180, 360, is_main=False, dist=540.0)
    c_panned = _make_candidate(650, 200, is_main=True, dist=20.0)
    scans = [
        # Post-zoom initial scan: main then side, with top edge clipped (main_bbox leaves empty top-left/right bg)
        TreeScanResult([c_main, c_side], [c_main, c_side], (380, 200, 480, 420), {"top"}, tree_coverage_ratio=0.35),
        TreeScanResult([c_side], [c_side], (380, 200, 480, 420), {"top"}, tree_coverage_ratio=0.35),
        TreeScanResult([], [], (380, 200, 480, 420), {"top"}, tree_coverage_ratio=0.35),
        # After panning top: discovers one more upgrade, then empty
        TreeScanResult([c_panned], [c_panned], (380, 260, 480, 420), set(), tree_coverage_ratio=0.30),
        TreeScanResult([], [], (380, 260, 480, 420), set(), tree_coverage_ratio=0.30),
        TreeScanResult([], [], (380, 200, 480, 420), {"top"}, tree_coverage_ratio=0.35),
    ]
    fsm.upgrade_scanner = lambda frame, **kwargs: scans.pop(0) if scans else TreeScanResult([], [], (380, 200, 480, 420), set(), tree_coverage_ratio=0.30)  # type: ignore[assignment]

    close_hit = MatchResult(found=True, confidence=0.95, center=(640, 660), template_name="btn_upgrades_close")
    close_miss = MatchResult(found=False, confidence=0.10, template_name="btn_upgrades_close")
    # Open (1), post-zoom (1), post-click c_main (1), post-click c_side (1), pan check (1), post-click c_panned (1), pan-back (1), close check (1), close confirm (miss)
    vision.match_seq["btn_upgrades_close"] = [close_hit] * 8 + [close_miss]

    fsm._transition_to(MacroState.UPGRADING)
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.READY_TO_START
    assert fsm.stats.upgrades_purchased == 3
    assert ctrl.clicks == [(640, 340, False), (1180, 360, False), (650, 200, False), (640, 660, False)]
    # Empty-background drag out and back in reverse direction
    assert len(ctrl.drags) == 2
    assert (ctrl.drags[1][0], ctrl.drags[1][1], ctrl.drags[1][2], ctrl.drags[1][3]) == (
        ctrl.drags[0][2],
        ctrl.drags[0][3],
        ctrl.drags[0][0],
        ctrl.drags[0][1],
    )


def test_auto_upgrade_blacklist_max_purchases_timeout_disconnect_and_esc_fallback() -> None:
    # 1. Blacklist stuck coordinate after 2 failed clicks & respect max_purchases_per_pass
    fsm, vision, ctrl, _, _ = _build_fsm()
    _enable_auto_upgrade(fsm, vision)
    fsm.config.auto_upgrade.max_purchases_per_pass = 1
    stuck = _make_candidate(500, 300, is_main=True, dist=5.0)
    good = _make_candidate(600, 300, is_main=True, dist=20.0)
    scans = [
        TreeScanResult([stuck, good], [stuck, good], None, set()),
        TreeScanResult([stuck, good], [stuck, good], None, set()),  # fail 1
        TreeScanResult([stuck, good], [stuck, good], None, set()),  # fail 2 -> blacklisted
        TreeScanResult([stuck], [stuck], None, set()),              # good consumed -> max_purchases=1 hit
    ]
    fsm.upgrade_scanner = lambda frame, **kwargs: scans.pop(0) if scans else TreeScanResult([], [], None, set())  # type: ignore[assignment]
    close_hit = MatchResult(found=True, confidence=0.95, center=(640, 660), template_name="btn_upgrades_close")
    close_miss = MatchResult(found=False, confidence=0.10, template_name="btn_upgrades_close")
    # Open (1), post-zoom (1), post-click stuck x2 (2), post-click good (1), close check (1), close confirm (miss)
    vision.match_seq["btn_upgrades_close"] = [close_hit] * 6 + [close_miss]

    fsm._transition_to(MacroState.UPGRADING)
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.READY_TO_START
    assert fsm.stats.upgrades_purchased == 1
    assert ctrl.clicks == [(500, 300, False), (500, 300, False), (600, 300, False), (640, 660, False)]

    # 2. pass_timeout_sec stops loop and closes via Esc fallback when close button stays visible
    fsm2, vision2, ctrl2, clk2, _ = _build_fsm()
    _enable_auto_upgrade(fsm2, vision2)
    fsm2.config.auto_upgrade.pass_timeout_sec = 5.0

    def _advancing_scanner(frame, **kwargs):
        clk2.advance(6.0)
        return TreeScanResult([good], [good], None, set())

    fsm2.upgrade_scanner = _advancing_scanner  # type: ignore[assignment]
    vision2.match_seq["btn_upgrades_close"] = [close_hit] * 8 + [close_miss]
    fsm2._transition_to(MacroState.UPGRADING)
    assert fsm2.step() == 0.0
    assert fsm2.state == MacroState.READY_TO_START
    assert fsm2.stats.upgrades_purchased == 0
    assert ("esc", False) in ctrl2.keys

    # 3. Mid-upgrade modal_disconnected aborts immediately to DISCONNECTED
    fsm3, vision3, ctrl3, _, _ = _build_fsm()
    _enable_auto_upgrade(fsm3, vision3)
    vision3.matches["btn_upgrades_close"] = close_hit
    vision3.matches["modal_disconnected"] = MatchResult(
        found=True, confidence=0.95, center=(640, 260), template_name="modal_disconnected"
    )
    fsm3._transition_to(MacroState.UPGRADING)
    assert fsm3.step() == 0.0
    assert fsm3.state == MacroState.DISCONNECTED
    assert ctrl3.clicks == []


def test_auto_upgrade_skipped_when_disabled_or_missing_templates_and_dry_run() -> None:
    # Disabled -> skips UPGRADING
    fsm, vision, ctrl, _, _ = _build_fsm()
    vision.loaded_optional.update({"btn_upgrades", "btn_upgrades_close"})
    fsm.config.auto_upgrade.enabled = False
    fsm._pending_upgrade_pass = True
    fsm._transition_to(MacroState.READY_TO_START)
    battle_hit = MatchResult(found=True, confidence=0.96, center=(680, 620), template_name="btn_battle")
    battle_miss = MatchResult(found=False, confidence=0.10, template_name="btn_battle")
    vision.match_seq["btn_battle"] = [battle_hit, battle_miss]
    fsm.step()
    assert fsm.state == MacroState.IN_RUN

    # Enabled, missing templates -> skips UPGRADING
    fsm2, vision2, _, _, _ = _build_fsm()
    fsm2.config.auto_upgrade.enabled = True
    fsm2._pending_upgrade_pass = True
    fsm2._transition_to(MacroState.READY_TO_START)
    vision2.match_seq["btn_battle"] = [battle_hit, battle_miss]
    fsm2.step()
    assert fsm2.state == MacroState.IN_RUN

    # Dry-Run mode scans open tree without clicking/scrolling/dragging
    fsm3, vision3, ctrl3, _, _ = _build_fsm()
    _enable_auto_upgrade(fsm3, vision3, scope="main_then_sides")
    fsm3.dry_run = True
    c_main = _make_candidate(540, 300, is_main=True)
    c_side = _make_candidate(1180, 360, is_main=False)
    fsm3.upgrade_scanner = lambda frame, **kwargs: TreeScanResult([c_main, c_side], [c_main, c_side], None, set())  # type: ignore[assignment]
    vision3.matches["btn_upgrades_close"] = MatchResult(
        found=True, confidence=0.95, center=(640, 660), template_name="btn_upgrades_close"
    )
    fsm3._transition_to(MacroState.READY_TO_START)
    assert fsm3.step() == 0.0
    assert fsm3.state == MacroState.UPGRADING
    delay = fsm3.step()
    assert delay == fsm3.config.polling.menu_interval_sec
    assert fsm3.state == MacroState.READY_TO_START
    assert ctrl3.clicks == []
    assert ctrl3.scrolls == []
    assert ctrl3.drags == []
    assert ctrl3.keys == []


def test_auto_upgrade_respects_instant_stop_event() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    _enable_auto_upgrade(fsm, vision)

    def _stop_on_click(client_x, client_y, hwnd=None, dry_run=False, park_cursor=True):
        ctrl.clicks.append((client_x, client_y, dry_run))
        fsm.stop_event.set()
        return True

    ctrl.click_client_point = _stop_on_click  # type: ignore[method-assign]
    c1 = _make_candidate(540, 300, is_main=True)
    fsm.upgrade_scanner = lambda frame, **kwargs: TreeScanResult([c1], [c1], None, set())  # type: ignore[assignment]
    vision.matches["btn_upgrades_close"] = MatchResult(
        found=True, confidence=0.95, center=(640, 660), template_name="btn_upgrades_close"
    )
    fsm._transition_to(MacroState.UPGRADING)
    assert fsm.step() is None
    assert fsm.stop_event.is_set()


def test_board_denylist_and_fail_closed_unreadable_never_click() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    _enable_auto_upgrade(fsm, vision, adaptive_zoom=False)

    board_tile = _make_candidate(420, 280, is_main=True, dist=10.0)
    unreadable_tile = _make_candidate(480, 280, is_main=True, dist=15.0)
    safe_tile = _make_candidate(560, 280, is_main=True, dist=20.0)

    def _ocr_for_tile(crop, **kwargs):
        marker = int(crop[0, 0, 0])
        if marker == 1:
            return ["Board V", "+1 Board Size", "$250.0M"]
        if marker == 2:
            return []  # Unreadable -> fail-closed
        return ["Damage XIII", "+30% Damage", "$100.0M"]

    hover_counter = {"idx": 0}

    def _fake_tooltip(before, after, pt):
        hover_counter["idx"] += 1
        if pt == (420, 280):
            return np.full((40, 120, 3), 1, dtype=np.uint8)
        if pt == (480, 280):
            return np.full((40, 120, 3), 2, dtype=np.uint8)
        return np.full((40, 120, 3), 3, dtype=np.uint8)

    fsm._ocr_engine = _ocr_for_tile
    fsm.tooltip_extractor = _fake_tooltip

    scans = [
        TreeScanResult([board_tile, unreadable_tile, safe_tile], [board_tile, unreadable_tile, safe_tile], None, set()),
        # After clicking safe_tile -> gone
        TreeScanResult([board_tile, unreadable_tile], [board_tile, unreadable_tile], None, set()),
    ]
    fsm.upgrade_scanner = lambda frame, **kwargs: scans.pop(0) if scans else TreeScanResult([], [], None, set())  # type: ignore[assignment]

    close_hit = MatchResult(found=True, confidence=0.95, center=(640, 660), template_name="btn_upgrades_close")
    close_miss = MatchResult(found=False, confidence=0.10, template_name="btn_upgrades_close")
    # Open (1), post-zoom (1), post-click safe_tile (1), close check (1), confirm close (miss)
    vision.match_seq["btn_upgrades_close"] = [close_hit] * 4 + [close_miss]

    fsm._transition_to(MacroState.UPGRADING)
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.READY_TO_START
    assert fsm.stats.upgrades_purchased == 1
    # Only safe_tile (560, 280) and btn_upgrades_close (640, 660) were ever clicked!
    assert ctrl.clicks == [(560, 280, False), (640, 660, False)]
    # Check logged messages include Blocked upgrade "Board V" and Purchased upgrade "Damage XIII"
    logs = []
    assert fsm.event_queue is not None
    while not fsm.event_queue.empty():
        ev = fsm.event_queue.get_nowait()
        if ev.get("type") == "log":
            logs.append(ev["message"])
    assert any('Blocked upgrade "Board V"' in m for m in logs)
    assert any('Purchased upgrade "Damage XIII"' in m for m in logs)


def test_adaptive_zoom_recentering_and_zero_pan_fast_path() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    _enable_auto_upgrade(fsm, vision, adaptive_zoom=True)

    # First scan: Start anchor off-center and too tall (height=65 > 44) and clipped
    # Second scan: After zoom out -2, Start height=38 (OK), off-center at (500, 345)
    # Third scan: After empty-background drag, centered at (640, 345), unclipped -> zero-pan fast path!
    scans = [
        TreeScanResult([], [], (280, 120, 600, 460), {"top"}, start_anchor=StartAnchor((500, 345), (470, 312, 60, 65), 65), tree_coverage_ratio=0.40),
        TreeScanResult([], [], (280, 140, 560, 420), set(), start_anchor=StartAnchor((500, 345), (482, 326, 36, 38), 38), tree_coverage_ratio=0.25),
        TreeScanResult([], [], (380, 140, 560, 420), set(), start_anchor=StartAnchor((640, 345), (622, 326, 36, 38), 38), tree_coverage_ratio=0.25),
    ]
    fsm.upgrade_scanner = lambda frame, **kwargs: scans.pop(0) if scans else TreeScanResult([], [], (380, 140, 560, 420), set())  # type: ignore[assignment]

    close_hit = MatchResult(found=True, confidence=0.95, center=(640, 660), template_name="btn_upgrades_close")
    close_miss = MatchResult(found=False, confidence=0.10, template_name="btn_upgrades_close")
    vision.match_seq["btn_upgrades_close"] = [close_hit] * 6 + [close_miss]

    fsm._transition_to(MacroState.UPGRADING)
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.READY_TO_START
    # Zoomed on Start anchor center (500, 345) by -2 notches, dragged once to center, zero edge-pan drags
    assert ctrl.scrolls == [(500, 345, -2, False)]
    assert len(ctrl.drags) == 1


def test_strict_priority_flow_spawners_before_upgrades_then_battle() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    _enable_auto_spawners(fsm, vision, every_n_runs=1, run_on_start=True, verify_buy_text_ocr=False)
    _enable_auto_upgrade(fsm, vision, every_n_runs=1, run_on_start=True, adaptive_zoom=False)

    fsm.spawner_scanner = lambda frame: SpawnerScanResult([], (370, 140, 896, 660))  # type: ignore[assignment]
    fsm.upgrade_scanner = lambda frame, **kwargs: TreeScanResult([], [], None, set())  # type: ignore[assignment]

    battle_hit = MatchResult(found=True, confidence=0.96, center=(680, 620), template_name="btn_battle")
    battle_miss = MatchResult(found=False, confidence=0.10, template_name="btn_battle")
    sp_hit = MatchResult(found=True, confidence=0.95, center=(1200, 580), template_name="btn_spawners")
    sp_close_hit = MatchResult(found=True, confidence=0.95, center=(900, 90), template_name="btn_spawners_close")
    sp_close_miss = MatchResult(found=False, confidence=0.10, template_name="btn_spawners_close")
    up_hit = MatchResult(found=True, confidence=0.95, center=(1200, 650), template_name="btn_upgrades")
    up_close_hit = MatchResult(found=True, confidence=0.95, center=(640, 660), template_name="btn_upgrades_close")
    up_close_miss = MatchResult(found=False, confidence=0.10, template_name="btn_upgrades_close")

    fsm._pending_spawner_pass = True
    fsm._pending_upgrade_pass = True
    fsm._transition_to(MacroState.READY_TO_START)

    # Tick 1: READY_TO_START sees Battle -> transitions to PURCHASING_SPAWNERS FIRST
    vision.matches["btn_battle"] = battle_hit
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.PURCHASING_SPAWNERS

    # Tick 2: PURCHASING_SPAWNERS runs and returns to READY_TO_START
    vision.matches["btn_spawners"] = sp_hit
    vision.match_seq["btn_spawners_close"] = [sp_close_miss, sp_close_hit, sp_close_hit, sp_close_miss]
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.READY_TO_START

    # Tick 3: READY_TO_START sees Battle -> now triggers UPGRADING SECOND
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.UPGRADING

    # Tick 4: UPGRADING runs and returns to READY_TO_START
    vision.matches["btn_upgrades"] = up_hit
    vision.match_seq["btn_upgrades_close"] = [up_close_miss, up_close_hit, up_close_hit, up_close_hit, up_close_miss]
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.READY_TO_START

    # Tick 5: Both passes done -> READY_TO_START clicks Battle -> IN_RUN
    vision.match_seq["btn_battle"] = [battle_hit, battle_miss]
    fsm.step()
    assert fsm.state == MacroState.IN_RUN
    assert ctrl.clicks[-1] == (680, 620, False)

    # Tick 6: Run completes -> schedules BOTH spawner and upgrade passes for next loop
    vision.matches["btn_battle"] = battle_hit
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.READY_TO_START
    assert fsm.stats.runs_completed == 1
    assert fsm._pending_spawner_pass is True
    assert fsm._pending_upgrade_pass is True


class _FakeShop:
    """Simulated Spawner Shop: per-card stock/price/page, a cash balance, and clicks Roblox may ignore."""

    ROI = (370, 140, 896, 660)

    def __init__(self, fsm, vision, ctrl, cards, cash, ignore_clicks=0):
        # cards: {pill_center: [stock, price, page]}
        self.cards = {c: list(v) for c, v in cards.items()}
        self.cash = cash
        self.page = 0
        self.ignore = ignore_clicks
        self.bought: list[tuple[int, int]] = []
        self.max_page = max(v[2] for v in self.cards.values())

        def click(client_x, client_y, hwnd=None, dry_run=False, park_cursor=True):
            ctrl.clicks.append((client_x, client_y, dry_run))
            card = self.cards.get((client_x, client_y))
            if card and card[2] == self.page and card[0] > 0:
                if self.ignore > 0:
                    self.ignore -= 1  # Roblox dropped the click
                elif self.cash >= card[1]:
                    card[0] -= 1
                    self.cash -= card[1]
                    self.bought.append((client_x, client_y))
            return True

        def scroll(client_x, client_y, notches, hwnd=None, dry_run=False):
            ctrl.scrolls.append((client_x, client_y, notches, dry_run))
            self.page = min(self.max_page, self.page + 1)
            return True

        ctrl.click_client_point = click  # type: ignore[method-assign]
        ctrl.scroll_client_point = scroll  # type: ignore[method-assign]
        vision.capture_client = lambda rect: np.full((720, 1280, 3), 40 + 40 * self.page, dtype=np.uint8)  # type: ignore[method-assign]
        fsm.spawner_scanner = lambda frame: SpawnerScanResult(  # type: ignore[assignment]
            [
                SpawnerBuyButton(center=c, bbox=(c[0] - 51, c[1] - 14, 102, 28), row_index=0, col_index=0, fill_ratio=0.85)
                for c, v in self.cards.items()
                if v[2] == self.page and v[0] > 0
            ],
            self.ROI,
        )
        fsm._read_spawner_stock = lambda frame, b: self.cards[b.center][0] if b.center in self.cards else None  # type: ignore[method-assign]
        fsm._get_ocr_lines = lambda crop, **kwargs: ["Buy 1,000"]  # type: ignore[method-assign]


def test_spawner_clicks_every_buy_cheapest_first_until_sold_out_then_stops_when_broke() -> None:
    """Click-all strategy: every green pill each round (row-major = cheapest first), multi-stock items
    keep getting clicked until sold out, and pills still green after the round cap end the pass."""
    fsm, vision, ctrl, _, _ = _build_fsm()
    _enable_auto_spawners(fsm, vision)
    fsm.config.auto_spawner.clicks_per_pill = 5
    _dynamic_close_x(vision, ctrl, (900, 90), flicker_calls=set())
    a, b, c = (446, 354), (575, 354), (446, 550)
    shop = _FakeShop(fsm, vision, ctrl, {a: [3, 1000, 0], b: [1, 1800, 0], c: [2, 50000, 1]}, cash=5000)

    fsm._transition_to(MacroState.PURCHASING_SPAWNERS)
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.READY_TO_START
    # round 1: a then b (left to right); rounds 2-3: a until sold out
    assert shop.bought == [a, b, a, a]
    clicked = [(x, y) for x, y, _ in ctrl.clicks]
    assert clicked[:4] == [a, b, a, a]
    # page 2: c is unaffordable -> clicked once per round, 5 rounds, then the pass stops (no more scrolling)
    assert clicked.count(c) == 5
    assert len(ctrl.scrolls) == 1
    assert clicked[-1] == (900, 90)
    assert fsm.stats.spawners_purchased == 9  # counts Buy clicks
    # every click is preceded by a hover on the same pill
    assert [(x, y) for x, y, _ in ctrl.hovers] == clicked[:-1]


def test_spawner_dropped_click_is_simply_retried_next_round() -> None:
    """High ping / dropped clicks: no confirmation needed, the still-green pill is clicked again."""
    fsm, vision, ctrl, _, _ = _build_fsm()
    _enable_auto_spawners(fsm, vision)
    _dynamic_close_x(vision, ctrl, (900, 90), flicker_calls=set())
    a = (446, 354)
    shop = _FakeShop(fsm, vision, ctrl, {a: [2, 1000, 0]}, cash=10_000, ignore_clicks=1)

    fsm._transition_to(MacroState.PURCHASING_SPAWNERS)
    fsm.step()
    assert shop.bought == [a, a]
    assert [(x, y) for x, y, _ in ctrl.clicks].count(a) == 3
    assert fsm.state == MacroState.READY_TO_START


def test_spawner_scrolls_through_pages_and_stops_at_end_of_list() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    _enable_auto_spawners(fsm, vision)
    fsm.config.auto_spawner.max_scroll_steps = 15
    _dynamic_close_x(vision, ctrl, (900, 90), flicker_calls=set())
    a, b = (446, 354), (446, 550)
    shop = _FakeShop(fsm, vision, ctrl, {a: [1, 1000, 0], b: [1, 1000, 1]}, cash=10_000)

    fsm._transition_to(MacroState.PURCHASING_SPAWNERS)
    fsm.step()
    assert shop.bought == [a, b]
    # page 1 -> page 2, then two scrolls that change nothing end the pass
    assert len(ctrl.scrolls) == 3
    assert fsm.state == MacroState.READY_TO_START


def test_spawner_scroll_end_stops_after_two_unchanged_scrolls_and_safety() -> None:
    # 1. Scroll end detection ignores background motion and stops after 2 unchanged scroll steps
    fsm, vision, ctrl, _, _ = _build_fsm()
    _enable_auto_spawners(fsm, vision, verify_buy_text_ocr=False)
    fsm.config.auto_spawner.max_scroll_steps = 10

    f_base = np.full((720, 1280, 3), 45, dtype=np.uint8)
    f_bg_animated = f_base.copy()
    f_bg_animated[20:100, 20:150] = 220  # outside grid ROI -> still unchanged!
    vision.frame_seq = [f_base, f_bg_animated, f_base]
    fsm.spawner_scanner = lambda frame: SpawnerScanResult([], (370, 140, 896, 660))  # type: ignore[assignment]

    sp_close_hit = MatchResult(found=True, confidence=0.95, center=(900, 90), template_name="btn_spawners_close")
    sp_close_miss = MatchResult(found=False, confidence=0.10, template_name="btn_spawners_close")
    vision.match_seq["btn_spawners_close"] = [sp_close_hit] * 5 + [sp_close_miss]

    fsm._transition_to(MacroState.PURCHASING_SPAWNERS)
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.READY_TO_START
    assert len(ctrl.scrolls) == 2

    # 2. Dry-Run scans open spawner shop without clicking or scrolling
    fsm2, vision2, ctrl2, _, _ = _build_fsm()
    _enable_auto_spawners(fsm2, vision2)
    fsm2.dry_run = True
    btn = SpawnerBuyButton(center=(446, 354), bbox=(395, 340, 102, 28), row_index=0, col_index=0, fill_ratio=0.85)
    fsm2.spawner_scanner = lambda frame: SpawnerScanResult([btn], (370, 140, 896, 660))  # type: ignore[assignment]
    vision2.matches["btn_spawners_close"] = sp_close_hit
    fsm2._transition_to(MacroState.READY_TO_START)
    assert fsm2.step() == 0.0
    assert fsm2.state == MacroState.PURCHASING_SPAWNERS
    assert fsm2.step() == fsm2.config.polling.menu_interval_sec
    assert fsm2.state == MacroState.READY_TO_START
    assert ctrl2.clicks == []
    assert ctrl2.scrolls == []

    # 3. Disconnect modal mid-spawner pass aborts cleanly to DISCONNECTED
    fsm3, vision3, ctrl3, _, _ = _build_fsm()
    _enable_auto_spawners(fsm3, vision3)
    vision3.matches["btn_spawners_close"] = sp_close_hit
    vision3.matches["modal_disconnected"] = MatchResult(
        found=True, confidence=0.95, center=(640, 260), template_name="modal_disconnected"
    )
    fsm3._transition_to(MacroState.PURCHASING_SPAWNERS)
    assert fsm3.step() == 0.0
    assert fsm3.state == MacroState.DISCONNECTED
    assert ctrl3.clicks == []


def test_run_ending_via_hud_loss_counts_run_and_queues_spawners_before_upgrades() -> None:
    """Regression: end-of-run screen hides the wave HUD before Battle reappears.

    IN_RUN falls back to READY_TO_START via the HUD-missing path; when Battle then shows,
    the run must be counted and both passes run (spawners first) before Battle is clicked.
    """
    fsm, vision, ctrl, _, _ = _build_fsm()
    _enable_auto_spawners(fsm, vision, every_n_runs=1, run_on_start=False)
    _enable_auto_upgrade(fsm, vision, every_n_runs=1, run_on_start=False, adaptive_zoom=False)
    fsm._transition_to(MacroState.IN_RUN)

    hud_miss = MatchResult(found=False, confidence=0.20, template_name="hud_in_run")
    battle_hit = MatchResult(found=True, confidence=0.99, center=(688, 603), template_name="btn_battle")
    battle_miss = MatchResult(found=False, confidence=0.19, template_name="btn_battle")
    vision.matches["hud_in_run"] = hud_miss
    vision.matches["btn_battle"] = battle_miss

    fsm.step()
    fsm.step()
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.READY_TO_START
    assert fsm.stats.runs_completed == 0

    # Results screen: nothing actionable yet -> stays waiting, no clicks
    fsm.step()
    assert fsm.state == MacroState.READY_TO_START
    assert ctrl.clicks == []

    # Lobby Battle/Continue appears -> run counted, spawner pass FIRST (no Battle click)
    vision.matches["btn_battle"] = battle_hit
    assert fsm.step() == 0.0
    assert fsm.stats.runs_completed == 1
    assert fsm.state == MacroState.PURCHASING_SPAWNERS
    assert fsm._pending_upgrade_pass is True
    assert ctrl.clicks == []


def test_mid_run_hud_dropout_does_not_count_run_when_hud_returns() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    _enable_auto_spawners(fsm, vision, every_n_runs=1, run_on_start=False)
    fsm._transition_to(MacroState.IN_RUN)

    hud_miss = MatchResult(found=False, confidence=0.20, template_name="hud_in_run")
    hud_hit = MatchResult(found=True, confidence=1.0, center=(820, 620), template_name="hud_in_run")
    battle_hit = MatchResult(found=True, confidence=0.99, center=(688, 603), template_name="btn_battle")
    battle_miss = MatchResult(found=False, confidence=0.19, template_name="btn_battle")
    vision.matches["hud_in_run"] = hud_miss
    vision.matches["btn_battle"] = battle_miss
    for _ in range(3):
        fsm.step()
    assert fsm.state == MacroState.READY_TO_START

    # Milestone banner clears -> HUD back -> IN_RUN, flag cleared
    vision.matches["hud_in_run"] = hud_hit
    fsm.step()
    assert fsm.state == MacroState.IN_RUN
    assert fsm._run_end_unconfirmed is False

    # Later a normal run end is counted exactly once
    vision.matches["hud_in_run"] = hud_miss
    vision.matches["btn_battle"] = battle_hit
    assert fsm.step() == 0.0
    assert fsm.stats.runs_completed == 1
    assert ctrl.clicks == []


def _dynamic_close_x(vision: FakeVision, ctrl: FakeController, close_center: tuple[int, int], flicker_calls: set[int]):
    """Close-X visible until it is clicked; hidden on the given call indexes (purchase overlay flicker)."""
    original_find = vision.find
    calls = {"n": 0}

    def find(frame, name, threshold=0.85, roi=None):
        if name != "btn_spawners_close":
            return original_find(frame, name, threshold, roi)
        calls["n"] += 1
        clicked_close = any((x, y) == close_center for x, y, _ in ctrl.clicks)
        if clicked_close or calls["n"] in flicker_calls:
            return MatchResult(found=False, confidence=0.10, template_name=name)
        return MatchResult(found=True, confidence=0.95, center=close_center, template_name=name)

    vision.find = find  # type: ignore[method-assign]
    return calls


def test_spawner_pass_survives_close_x_flicker_after_purchase() -> None:
    """Regression: a purchase overlay hid the close X for a frame; the pass must keep buying
    and close the shop itself instead of ending early with the shop left open."""
    fsm, vision, ctrl, _, _ = _build_fsm()
    _enable_auto_spawners(fsm, vision, every_n_runs=1, run_on_start=False, verify_buy_text_ocr=True)
    close_center = (900, 90)
    # Call 1: open check (visible). Call 2: first post-click poll after btn A -> hidden by overlay.
    _dynamic_close_x(vision, ctrl, close_center, flicker_calls={2})
    a, b = (822, 347), (568, 584)
    shop = _FakeShop(fsm, vision, ctrl, {a: [1, 1000, 0], b: [1, 1000, 0]}, cash=10_000)

    fsm._transition_to(MacroState.PURCHASING_SPAWNERS)
    assert fsm.step() == 0.0
    assert fsm.state == MacroState.READY_TO_START
    assert shop.bought == [a, b]
    assert fsm.stats.spawners_purchased == 2
    clicked = [(x, y) for x, y, _ in ctrl.clicks]
    assert clicked[:2] == [a, b]
    assert clicked[-1] == close_center


def test_ready_to_start_closes_spawner_shop_left_open() -> None:
    fsm, vision, ctrl, _, _ = _build_fsm()
    _enable_auto_spawners(fsm, vision, every_n_runs=1, run_on_start=False)
    close_center = (900, 90)
    _dynamic_close_x(vision, ctrl, close_center, flicker_calls=set())
    vision.matches["btn_battle"] = MatchResult(found=False, confidence=0.20, template_name="btn_battle")

    fsm._transition_to(MacroState.READY_TO_START)
    fsm.step()
    assert fsm.state == MacroState.READY_TO_START
    assert [(x, y) for x, y, _ in ctrl.clicks] == [close_center]


def test_spawner_shop_open_retries_when_first_click_is_ignored() -> None:
    """Regression: right after Roblox regained focus the first Spawners click was ignored and the
    whole pass was skipped. The open step must hover and retry."""
    fsm, vision, ctrl, _, _ = _build_fsm()
    _enable_auto_spawners(fsm, vision, every_n_runs=1, run_on_start=False)
    fsm.config.polling.click_confirm_timeout_sec = 0.002
    open_center = (786, 600)
    vision.matches["btn_spawners"] = MatchResult(found=True, confidence=0.91, center=open_center, template_name="btn_spawners")
    state = {"open": False, "open_clicks": 0}
    real_click = ctrl.click_client_point

    def click(client_x, client_y, hwnd=None, dry_run=False, park_cursor=True):
        if (client_x, client_y) == open_center:
            state["open_clicks"] += 1
            if state["open_clicks"] >= 2:  # first click dropped
                state["open"] = True
        if (client_x, client_y) == (900, 90):
            state["open"] = False
        return real_click(client_x, client_y, hwnd=hwnd, dry_run=dry_run, park_cursor=park_cursor)

    ctrl.click_client_point = click  # type: ignore[method-assign]
    original_find = vision.find

    def find(frame, name, threshold=0.85, roi=None):
        if name == "btn_spawners_close":
            if state["open"]:
                return MatchResult(found=True, confidence=0.99, center=(900, 90), template_name=name)
            return MatchResult(found=False, confidence=0.1, template_name=name)
        return original_find(frame, name, threshold, roi)

    vision.find = find  # type: ignore[method-assign]
    fsm.spawner_scanner = lambda frame: SpawnerScanResult([], (370, 140, 896, 660))  # type: ignore[assignment]

    fsm._transition_to(MacroState.PURCHASING_SPAWNERS)
    fsm.step()
    assert state["open_clicks"] == 2
    assert ctrl.hovers[:2] == [(786, 600, False), (786, 600, False)]
    assert (900, 90, False) in ctrl.clicks  # shop was opened and then closed by the pass
    assert fsm.state == MacroState.READY_TO_START


def test_ready_to_start_refocuses_roblox_when_covered() -> None:
    fsm, vision, ctrl, clk, _ = _build_fsm()
    focused = {"value": False}
    ctrl.is_focused = lambda hwnd=None: focused["value"]  # type: ignore[attr-defined]
    fsm._transition_to(MacroState.READY_TO_START)
    vision.matches["btn_battle"] = MatchResult(found=False, confidence=0.2, template_name="btn_battle")

    base_focus = ctrl.focus_calls
    fsm.step()
    assert ctrl.focus_calls == base_focus + 1
    fsm.step()  # throttled: no second focus within 5s
    assert ctrl.focus_calls == base_focus + 1
    clk.advance(6.0)
    focused["value"] = True
    fsm.step()  # already foreground -> no focus
    assert ctrl.focus_calls == base_focus + 1
    assert ctrl.clicks == []


def test_spawner_stops_scrolling_after_three_empty_pages_despite_animation() -> None:
    """Regression: animated cards made every scroll look 'changed', so the pass scrolled all 15 steps
    at the bottom of the list. Three empty pages in a row now end it."""
    fsm, vision, ctrl, _, _ = _build_fsm()
    _enable_auto_spawners(fsm, vision)
    fsm.config.auto_spawner.max_scroll_steps = 15
    _dynamic_close_x(vision, ctrl, (900, 90), flicker_calls=set())
    a = (446, 354)
    shop = _FakeShop(fsm, vision, ctrl, {a: [1, 1000, 0]}, cash=10_000)
    tick = {"n": 0}

    def animated_frame(rect):
        tick["n"] += 1
        f = np.full((720, 1280, 3), 40, dtype=np.uint8)
        f[100:700, 300:1000] = (tick["n"] * 37) % 255  # whole shop grid animates
        return f

    vision.capture_client = animated_frame  # type: ignore[method-assign]
    fsm._transition_to(MacroState.PURCHASING_SPAWNERS)
    fsm.step()
    assert shop.bought == [a]
    assert len(ctrl.scrolls) == 3
