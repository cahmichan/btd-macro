"""Tests for configuration defaults, partial JSON merging, and Win32 struct safety."""

from __future__ import annotations

import ctypes
import json
import re
from pathlib import Path

import win32con
import win32gui
import win32process

from src.config import AppConfig, load_config, save_config
from src.win32_utils import INPUT, kernel32, resolve_virtual_key, user32


def test_default_config_values() -> None:
    cfg = AppConfig()
    assert cfg.place_id == "113140456364856"
    assert (cfg.window.x, cfg.window.y, cfg.window.width, cfg.window.height) == (0, 0, 1280, 720)
    assert cfg.thresholds.btn_battle == 0.85
    assert cfg.thresholds.modal_disconnected == 0.85
    assert cfg.thresholds.btn_reconnect == 0.85
    assert cfg.thresholds.btn_yes_continue == 0.85
    assert cfg.thresholds.btn_continue == 0.85
    assert cfg.thresholds.btn_start_over == 0.85
    assert cfg.thresholds.hud_in_run == 0.85
    assert cfg.rois.btn_battle == [0.42, 0.75, 0.70, 1.0]
    assert cfg.rois.btn_yes_continue == [0.30, 0.42, 0.72, 0.72]
    assert cfg.rois.btn_continue == [0.42, 0.75, 0.70, 1.0]
    assert cfg.rois.hud_in_run == [0.22, 0.74, 0.78, 1.0]
    assert cfg.anti_afk.interval_sec == 720.0
    assert cfg.recovery.max_reconnect_retries == 2
    assert cfg.recovery.circuit_breaker_max_triggers == 5
    assert cfg.recovery.circuit_breaker_window_sec == 900.0
    assert cfg.recovery.resume_mode == "continue"
    assert cfg.logging.max_failure_snapshots == 20
    assert cfg.hotkey == "F8"


def test_config_roundtrip_and_auto_create(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    created = load_config(cfg_path)
    assert cfg_path.is_file()
    assert created.place_id == "113140456364856"

    created.window.width = 1600
    created.thresholds.btn_battle = 0.91
    created.anti_afk.key = "w"
    created.recovery.resume_mode = "start_over"
    save_config(created, cfg_path)

    reloaded = load_config(cfg_path)
    assert reloaded.window.width == 1600
    assert reloaded.thresholds.btn_battle == 0.91
    assert reloaded.anti_afk.key == "w"
    assert reloaded.recovery.resume_mode == "start_over"
    assert reloaded.polling.in_run_interval_sec == 4.0


def test_partial_config_merges_onto_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "partial.json"
    cfg_path.write_text(
        json.dumps(
            {
                "place_id": "999",
                "window": {"x": 10},
                "thresholds": {"btn_battle": 0.78},
            }
        ),
        encoding="utf-8",
    )
    loaded = load_config(cfg_path)
    assert loaded.place_id == "999"
    assert loaded.window.x == 10
    assert loaded.window.width == 1280
    assert loaded.window.height == 720
    assert loaded.thresholds.btn_battle == 0.78
    assert loaded.thresholds.modal_disconnected == 0.85
    assert loaded.recovery.circuit_breaker_max_triggers == 5


def test_win32_sendinput_struct_size_and_vk_resolution() -> None:
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        assert ctypes.sizeof(INPUT) == 40
    assert resolve_virtual_key("space") == 0x20
    assert resolve_virtual_key("F8") == 0x77
    assert resolve_virtual_key("w") == ord("W")


def test_win32_utils_all_api_attributes_exist() -> None:
    src = Path(__file__).resolve().parent.parent / "src" / "win32_utils.py"
    text = src.read_text(encoding="utf-8")
    modules = {
        "win32gui": win32gui,
        "win32con": win32con,
        "win32process": win32process,
        "user32": user32,
        "kernel32": kernel32,
    }
    for mod_name, mod in modules.items():
        attrs = sorted(set(re.findall(rf"\b{mod_name}\.([A-Za-z_][A-Za-z0-9_]*)", text)))
        missing = [attr for attr in attrs if not hasattr(mod, attr)]
        assert missing == [], f"Missing attributes on {mod_name}: {missing}"
