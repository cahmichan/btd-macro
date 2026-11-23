"""Unit and regression tests for upgrade tree OpenCV clustering and affordable tile detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.upgrade_vision import (
    UpgradeCandidate,
    detect_affordable_upgrades,
    is_same_candidate,
)

# Real screenshot fixtures in parent workspace folder (skipped cleanly if absent)
SCREENSHOT_DIR = Path(__file__).resolve().parent.parent.parent
SHOT_MID_ZOOM = SCREENSHOT_DIR / "RobloxScreenShot20261123_094905285.png"
SHOT_ZOOMED_IN = SCREENSHOT_DIR / "RobloxScreenShot20261123_094851156.png"

CREAM_YELLOW_BGR = (127, 220, 249)  # RGB(249, 220, 127)
CORAL_RED_BGR = (101, 117, 226)     # RGB(226, 117, 101)
AMBER_BORDER_BGR = (40, 175, 240)   # warm amber/gold border


def _draw_synthetic_tile(
    canvas: np.ndarray,
    cx: int,
    cy: int,
    kind: str,
    price_text: str = "$50.0M",
) -> None:
    """Draw a synthetic hex-like upgrade tile centered near (cx, cy)."""
    r = 26
    pts = np.array(
        [
            [cx, cy - r],
            [cx + r, cy - r // 2],
            [cx + r, cy + r // 2],
            [cx, cy + r],
            [cx - r, cy + r // 2],
            [cx - r, cy - r // 2],
        ],
        dtype=np.int32,
    )

    if kind == "unlocked":
        # Bright gold unlocked interior, black border, no price row
        cv2.fillPoly(canvas, [pts], (60, 210, 235))
        cv2.polylines(canvas, [pts], True, (15, 15, 15), 2)
        cv2.putText(canvas, "Cash", (cx - 16, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        return

    if kind == "mystery":
        # Dark gray interior, amber border, white '?', no price row
        cv2.fillPoly(canvas, [pts], (55, 55, 55))
        cv2.polylines(canvas, [pts], True, AMBER_BORDER_BGR, 3)
        cv2.putText(canvas, "?", (cx - 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        return

    if kind == "toggle":
        # Bright blue/gold toggle tile ('START 50') with no cream-yellow price row
        cv2.fillPoly(canvas, [pts], (200, 140, 50))
        cv2.polylines(canvas, [pts], True, AMBER_BORDER_BGR, 3)
        cv2.putText(canvas, "START 50", (cx - 22, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (245, 245, 245), 1)
        return

    # Dimmed interior with colored upper icon + amber border
    cv2.fillPoly(canvas, [pts], (45, 48, 52))
    cv2.circle(canvas, (cx, cy - 8), 9, (50, 150, 90), -1)
    cv2.polylines(canvas, [pts], True, AMBER_BORDER_BGR, 3)

    text_color = CREAM_YELLOW_BGR if kind == "affordable" else CORAL_RED_BGR
    cv2.putText(
        canvas,
        price_text,
        (cx - 19, cy + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.33,
        text_color,
        1,
        cv2.LINE_AA,
    )


def test_detect_affordable_filters_colors_hud_and_prioritizes_main_tree() -> None:
    frame = np.full((720, 1280, 3), 18, dtype=np.uint8)

    # Central main tree cluster (~centered around 640, 360) with adjacent unlocked/affordable/unaffordable/?/toggle
    _draw_synthetic_tile(frame, 640, 360, "unlocked")
    _draw_synthetic_tile(frame, 694, 360, "unlocked")
    _draw_synthetic_tile(frame, 586, 360, "affordable", "$25.0M")   # main inner (closer to center)
    _draw_synthetic_tile(frame, 748, 360, "unaffordable", "$1.50B")  # coral-red, ignored
    _draw_synthetic_tile(frame, 640, 306, "mystery")                 # '?', ignored
    _draw_synthetic_tile(frame, 640, 414, "toggle")                  # 'START 50', ignored
    _draw_synthetic_tile(frame, 532, 360, "affordable", "$50.0M")   # main outer

    # Disconnected side island far to the right
    _draw_synthetic_tile(frame, 1140, 360, "affordable", "$10.0M")

    # Fake cream-yellow text inside masked HUD areas (bottom-left cash panel & top header)
    cv2.putText(frame, "$999.9M", (60, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CREAM_YELLOW_BGR, 1)
    cv2.putText(frame, "Spend cash", (590, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CREAM_YELLOW_BGR, 1)

    res_main = detect_affordable_upgrades(frame, scope="main_only", order="center_first")
    assert res_main.clipped_edges == set()
    assert len(res_main.candidates) == 2
    assert all(c.is_main_tree for c in res_main.candidates)
    # Center-first puts 586 before 532 (closer to ~640 centroid)
    assert res_main.candidates[0].price_center[0] > res_main.candidates[1].price_center[0]
    assert res_main.candidates[0].click_point[1] < res_main.candidates[0].price_center[1]

    res_sides = detect_affordable_upgrades(frame, scope="main_then_sides", order="center_first")
    assert len(res_sides.candidates) == 3
    assert res_sides.candidates[0].is_main_tree is True
    assert res_sides.candidates[1].is_main_tree is True
    assert res_sides.candidates[2].is_main_tree is False
    assert res_sides.candidates[2].price_center[0] > 1050


def test_top_down_ordering_and_edge_clipping_flags() -> None:
    # Touch top edge (between top-left Roblox bar [0..204] and top-center header [460..819])
    frame_top = np.full((720, 1280, 3), 18, dtype=np.uint8)
    _draw_synthetic_tile(frame_top, 260, 34, "unlocked")
    _draw_synthetic_tile(frame_top, 314, 34, "affordable", "$10.0M")
    _draw_synthetic_tile(frame_top, 260, 88, "affordable", "$20.0M")
    _draw_synthetic_tile(frame_top, 368, 34, "unlocked")

    res_top = detect_affordable_upgrades(frame_top, scope="main_only", order="top_down")
    assert "top" in res_top.clipped_edges
    assert "bottom" not in res_top.clipped_edges
    assert "right" not in res_top.clipped_edges
    assert len(res_top.candidates) == 2
    # top_down orders smaller Y first (y~46 before y~100)
    assert res_top.candidates[0].price_center[1] < res_top.candidates[1].price_center[1]

    # Touch left edge below top-left bar (y=120) and above cash panel (y=367)
    frame_left = np.full((720, 1280, 3), 18, dtype=np.uint8)
    _draw_synthetic_tile(frame_left, 34, 150, "unlocked")
    _draw_synthetic_tile(frame_left, 88, 150, "affordable", "$15.0M")
    res_left = detect_affordable_upgrades(frame_left, scope="main_only")
    assert "left" in res_left.clipped_edges
    assert "top" not in res_left.clipped_edges

    # Touch bottom and right edges
    frame_br = np.full((720, 1280, 3), 18, dtype=np.uint8)
    _draw_synthetic_tile(frame_br, 1244, 684, "unlocked")
    _draw_synthetic_tile(frame_br, 1190, 684, "affordable", "$10.0M")
    res_br = detect_affordable_upgrades(frame_br, scope="main_only")
    assert "bottom" in res_br.clipped_edges
    assert "right" in res_br.clipped_edges


def test_is_same_candidate_and_empty_frame_safety() -> None:
    empty_res = detect_affordable_upgrades(None)
    assert empty_res.candidates == []
    assert empty_res.main_tree_bbox is None
    assert empty_res.clipped_edges == set()

    c1 = UpgradeCandidate((100, 114), (100, 100), (90, 110, 20, 8), True, 10.0, 30)
    c2 = UpgradeCandidate((110, 120), (112, 106), (100, 116, 20, 8), True, 12.0, 30)
    c3 = UpgradeCandidate((150, 114), (150, 100), (140, 110, 20, 8), True, 50.0, 30)

    assert is_same_candidate(c1, c2, tolerance_px=24) is True
    assert is_same_candidate(c1, (112, 106), tolerance_px=24) is True
    assert is_same_candidate(c1, c3, tolerance_px=24) is False


@pytest.mark.skipif(not SHOT_MID_ZOOM.is_file(), reason="Mid-zoom screenshot fixture not present")
def test_real_mid_zoom_screenshot_at_1280x720() -> None:
    img = cv2.imread(str(SHOT_MID_ZOOM))
    assert img is not None
    img720 = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)

    res_main = detect_affordable_upgrades(img720, scope="main_only")
    assert len(res_main.candidates) == 9
    assert res_main.clipped_edges == set()

    res_all = detect_affordable_upgrades(img720, scope="main_then_sides")
    assert len(res_all.candidates) == 11
    assert sum(1 for c in res_all.candidates if c.is_main_tree) == 9
    assert sum(1 for c in res_all.candidates if not c.is_main_tree) == 2


@pytest.mark.skipif(not SHOT_ZOOMED_IN.is_file(), reason="Zoomed-in screenshot fixture not present")
def test_real_zoomed_in_screenshot_clips_all_edges_at_1280x720() -> None:
    img = cv2.imread(str(SHOT_ZOOMED_IN))
    assert img is not None
    img720 = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)

    res = detect_affordable_upgrades(img720, scope="main_only")
    assert len(res.candidates) == 4
    assert res.clipped_edges == {"top", "bottom", "left", "right"}
