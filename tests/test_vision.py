"""Tests for BGR template matching, ROIs, dismiss templates, and snapshot rotation."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from src.vision import VisionEngine


def _make_textured_button(bgr_color: tuple[int, int, int], w: int = 80, h: int = 40) -> np.ndarray:
    """Create a synthetic textured button (non-uniform so TM_CCOEFF_NORMED is well-defined)."""
    btn = np.zeros((h, w, 3), dtype=np.uint8)
    btn[:, :] = bgr_color
    cv2.rectangle(btn, (4, 4), (w - 5, h - 5), (255, 255, 255), 2)
    cv2.putText(btn, "GO", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return btn


def test_bgr_template_matching_discriminates_color(tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    red_btn = _make_textured_button((40, 40, 230))
    blue_btn = _make_textured_button((230, 40, 40))

    cv2.imwrite(str(assets_dir / "btn_battle.png"), red_btn)
    engine = VisionEngine(assets_dir=assets_dir, dismiss_dir=tmp_path / "dismiss", failures_dir=tmp_path / "failures")

    frame = np.full((300, 400, 3), 25, dtype=np.uint8)
    frame[200:240, 180:260] = red_btn
    hit = engine.find(frame, "btn_battle", threshold=0.90)
    assert hit.found is True
    assert hit.confidence >= 0.99
    assert hit.center == (220, 220)
    assert hit.bbox == (180, 200, 80, 40)

    # Same shape/text but wrong color (blue instead of red) should score lower in BGR
    wrong_frame = np.full((300, 400, 3), 25, dtype=np.uint8)
    wrong_frame[200:240, 180:260] = blue_btn
    miss = engine.find(wrong_frame, "btn_battle", threshold=0.85)
    assert miss.found is False
    assert miss.confidence < 0.85


def test_roi_inclusion_exclusion_and_small_roi_guard() -> None:
    frame = np.full((400, 600, 3), 20, dtype=np.uint8)
    btn = _make_textured_button((30, 180, 60), w=60, h=30)
    frame[320:350, 300:360] = btn

    inside = VisionEngine.match_template(frame, btn, threshold=0.90, roi=[0.4, 0.7, 0.7, 1.0])
    assert inside.found is True
    assert inside.center == (330, 335)
    assert inside.bbox == (300, 320, 60, 30)

    outside = VisionEngine.match_template(frame, btn, threshold=0.85, roi=[0.0, 0.0, 0.4, 0.5])
    assert outside.found is False

    tiny_roi = VisionEngine.match_template(frame, btn, threshold=0.85, roi=[0.0, 0.0, 0.05, 0.05])
    assert tiny_roi.found is False
    assert tiny_roi.confidence == 0.0


def test_dismiss_templates_and_missing_required(tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets"
    dismiss_dir = assets_dir / "dismiss"
    dismiss_dir.mkdir(parents=True)

    cv2.imwrite(str(assets_dir / "btn_battle.png"), _make_textured_button((40, 40, 220)))
    cv2.imwrite(str(assets_dir / "hud_in_run.png"), _make_textured_button((30, 30, 210)))
    cv2.imwrite(str(assets_dir / "btn_yes_continue.png"), _make_textured_button((40, 180, 40)))
    cv2.imwrite(str(assets_dir / "btn_continue.png"), _make_textured_button((40, 40, 180)))
    cv2.imwrite(str(assets_dir / "btn_start_over.png"), _make_textured_button((120, 100, 80)))
    cv2.imwrite(str(assets_dir / "modal_same_account.png"), _make_textured_button((140, 140, 140)))
    cv2.imwrite(str(assets_dir / "btn_upgrades.png"), _make_textured_button((40, 160, 220)))
    cv2.imwrite(str(assets_dir / "btn_upgrades_close.png"), _make_textured_button((60, 60, 210)))
    cv2.imwrite(str(assets_dir / "btn_spawners.png"), _make_textured_button((40, 130, 200)))
    cv2.imwrite(str(assets_dir / "btn_spawners_close.png"), _make_textured_button((50, 50, 220)))
    popup_close = _make_textured_button((60, 60, 60), w=50, h=30)
    cv2.imwrite(str(dismiss_dir / "close_x.png"), popup_close)

    engine = VisionEngine(assets_dir=assets_dir, dismiss_dir=dismiss_dir, failures_dir=tmp_path / "failures")
    assert engine.missing_required_templates() == ["modal_disconnected", "btn_reconnect"]
    assert engine.has_template("hud_in_run") is True
    assert engine.has_template("btn_yes_continue") is True
    assert engine.has_template("btn_continue") is True
    assert engine.has_template("btn_start_over") is True
    assert engine.has_template("modal_same_account") is True
    assert engine.has_template("btn_upgrades") is True
    assert engine.has_template("btn_upgrades_close") is True
    assert engine.has_template("btn_spawners") is True
    assert engine.has_template("btn_spawners_close") is True

    frame = np.full((240, 320, 3), 15, dtype=np.uint8)
    frame[50:80, 120:170] = popup_close

    found = engine.find_dismiss_target(frame, threshold=0.85)
    assert found is not None
    assert found.found is True
    assert found.template_name == "dismiss/close_x"
    assert found.center == (145, 65)


def test_failure_snapshot_rotation_caps_at_limit(tmp_path: Path) -> None:
    failures_dir = tmp_path / "failures"
    engine = VisionEngine(
        assets_dir=tmp_path / "assets",
        dismiss_dir=tmp_path / "dismiss",
        failures_dir=failures_dir,
        max_failure_snapshots=20,
    )
    frame = np.full((120, 160, 3), 100, dtype=np.uint8)

    saved_paths: list[Path] = []
    base_ns = 1_700_000_000_000_000_000
    for idx in range(25):
        path = engine.save_failure_snapshot(frame, f"reason_{idx}")
        assert path is not None
        os.utime(path, ns=(base_ns + idx * 1_000_000, base_ns + idx * 1_000_000))
        saved_paths.append(path)

    # Re-prune after explicit monotonic mtimes to guarantee deterministic ordering
    engine._prune_failure_snapshots()
    remaining = sorted(failures_dir.glob("*.png"))
    assert len(remaining) == 20
    for old_path in saved_paths[:5]:
        assert not old_path.exists()
    for kept_path in saved_paths[5:]:
        assert kept_path.exists()
