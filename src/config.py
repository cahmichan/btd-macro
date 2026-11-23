"""Configuration management and project path helpers for the BTD overnight macro."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, TypeVar

T = TypeVar("T")


@dataclass
class WindowConfig:
    """Target Roblox window position and dimensions."""

    x: int = 0
    y: int = 0
    width: int = 1280
    height: int = 720
    size_mode: str = "window"  # "window" (outer frame) or "client" (client area)


@dataclass
class ThresholdConfig:
    """BGR normalized cross-correlation thresholds for template matching."""

    btn_battle: float = 0.85
    modal_disconnected: float = 0.85
    modal_same_account: float = 0.85
    btn_reconnect: float = 0.85
    btn_yes_continue: float = 0.85
    btn_continue: float = 0.85
    btn_start_over: float = 0.85
    hud_in_run: float = 0.85
    btn_upgrades: float = 0.85
    btn_upgrades_close: float = 0.85
    dismiss: float = 0.85


@dataclass
class RoiConfig:
    """Normalized [x1, y1, x2, y2] search regions within the Roblox client area."""

    btn_battle: list[float] = field(default_factory=lambda: [0.42, 0.75, 0.70, 1.0])
    modal_disconnected: list[float] = field(default_factory=lambda: [0.15, 0.15, 0.85, 0.85])
    modal_same_account: list[float] = field(default_factory=lambda: [0.15, 0.15, 0.85, 0.85])
    btn_reconnect: list[float] = field(default_factory=lambda: [0.15, 0.15, 0.85, 0.85])
    btn_yes_continue: list[float] = field(default_factory=lambda: [0.30, 0.42, 0.72, 0.72])
    btn_start_over: list[float] = field(default_factory=lambda: [0.30, 0.42, 0.72, 0.72])
    btn_continue: list[float] = field(default_factory=lambda: [0.42, 0.75, 0.70, 1.0])
    hud_in_run: list[float] = field(default_factory=lambda: [0.22, 0.74, 0.78, 1.0])
    btn_upgrades: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])
    btn_upgrades_close: list[float] = field(default_factory=lambda: [0.38, 0.85, 0.62, 1.0])


@dataclass
class PollingConfig:
    """Adaptive polling intervals and click confirmation timings."""

    menu_interval_sec: float = 1.0
    in_run_interval_sec: float = 4.0
    click_confirm_timeout_sec: float = 3.0
    click_confirm_poll_sec: float = 0.4
    post_launch_settle_sec: float = 2.0


@dataclass
class TimeoutConfig:
    """Per-state maximum duration watchdogs in seconds."""

    launching_sec: float = 180.0
    launching_update_grace_sec: float = 300.0
    ready_to_start_sec: float = 120.0
    upgrading_sec: float = 60.0
    in_run_sec: float = 2700.0
    disconnected_sec: float = 90.0


@dataclass
class AntiAfkConfig:
    """Anti-AFK settings to prevent Roblox's 20-minute idle disconnect."""

    enabled: bool = True
    interval_sec: float = 720.0
    method: str = "key_tap"  # "key_tap" or "mouse_nudge"
    key: str = "space"


@dataclass
class RecoveryConfig:
    """Disconnect retry limits and consecutive recovery circuit breaker settings."""

    max_reconnect_retries: int = 2
    reconnect_wait_sec: float = 10.0
    max_click_retries: int = 5
    circuit_breaker_max_triggers: int = 5
    circuit_breaker_window_sec: float = 900.0
    resume_mode: str = "continue"  # "continue" (Yes, continue) or "start_over" (Start over)
    stop_on_same_account: bool = True


@dataclass
class AutoUpgradeConfig:
    """Between-run and startup hex upgrade tree automation settings."""

    enabled: bool = False
    every_n_runs: int = 1
    run_on_start: bool = True
    scope: str = "main_only"  # "main_only" or "main_then_sides"
    order: str = "center_first"  # "center_first" or "top_down"
    zoom_reset_scrolls: int = 15
    zoom_in_scrolls: int = 5
    enable_edge_panning: bool = True
    pan_drag_px: int = 260
    max_purchases_per_pass: int = 25
    pass_timeout_sec: float = 45.0
    post_click_settle_sec: float = 0.35
    tile_click_y_offset_px: int = -14


@dataclass
class LoggingConfig:
    """Logging and failure snapshot retention settings."""

    max_failure_snapshots: int = 20


@dataclass
class UiConfig:
    """Control panel dimensions and docking offset beside the Roblox window."""

    panel_width: int = 380
    panel_height: int = 720
    panel_gap: int = 12


@dataclass
class AppConfig:
    """Top-level application configuration."""

    place_id: str = "113140456364856"
    window: WindowConfig = field(default_factory=WindowConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    rois: RoiConfig = field(default_factory=RoiConfig)
    polling: PollingConfig = field(default_factory=PollingConfig)
    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)
    anti_afk: AntiAfkConfig = field(default_factory=AntiAfkConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    auto_upgrade: AutoUpgradeConfig = field(default_factory=AutoUpgradeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    hotkey: str = "F8"


def get_project_root() -> Path:
    """Return the root directory of the btd-macro project."""
    return Path(__file__).resolve().parent.parent


def get_config_path() -> Path:
    """Return the default path to config.json."""
    return get_project_root() / "config.json"


def get_assets_dir() -> Path:
    """Return the path to the template assets directory."""
    return get_project_root() / "assets"


def get_dismiss_dir() -> Path:
    """Return the path to the optional dismiss templates directory."""
    return get_assets_dir() / "dismiss"


def get_logs_dir() -> Path:
    """Return the path to the session logs directory."""
    return get_project_root() / "logs"


def get_failures_dir() -> Path:
    """Return the path to the failure snapshots directory."""
    return get_logs_dir() / "failures"


def _merge_dataclass(cls: type[T], raw: Any) -> T:
    """Recursively instantiate a dataclass by merging partial dict overrides onto defaults."""
    default_instance = cls()
    if not isinstance(raw, dict):
        return default_instance

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        default_val = getattr(default_instance, f.name)
        if f.name not in raw:
            kwargs[f.name] = default_val
            continue

        incoming = raw[f.name]
        if is_dataclass(default_val) and isinstance(incoming, dict):
            kwargs[f.name] = _merge_dataclass(type(default_val), incoming)
        elif isinstance(default_val, list) and isinstance(incoming, list):
            kwargs[f.name] = [float(v) for v in incoming] if len(incoming) == 4 else default_val
        elif isinstance(default_val, bool):
            kwargs[f.name] = bool(incoming)
        elif isinstance(default_val, int) and not isinstance(default_val, bool):
            kwargs[f.name] = int(incoming)
        elif isinstance(default_val, float):
            kwargs[f.name] = float(incoming)
        elif isinstance(default_val, str):
            kwargs[f.name] = str(incoming)
        else:
            kwargs[f.name] = incoming
    return cls(**kwargs)


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load configuration from JSON, deep-merging onto defaults.

    Creates a default config.json if the target file does not exist.
    """
    cfg_path = Path(path) if path is not None else get_config_path()
    if not cfg_path.exists():
        cfg = AppConfig()
        save_config(cfg, cfg_path)
        return cfg

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return _merge_dataclass(AppConfig, raw)


def save_config(cfg: AppConfig, path: Optional[Path] = None) -> None:
    """Persist an AppConfig instance to disk as formatted JSON."""
    cfg_path = Path(path) if path is not None else get_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=2)
        fh.write("\n")
