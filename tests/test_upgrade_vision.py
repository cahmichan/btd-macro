"""Unit and regression tests for upgrade tree OpenCV clustering and affordable tile detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.upgrade_vision import (
    StartAnchor,
    UpgradeCandidate,
    build_hud_mask,
    classify_upgrade_tooltip,
    detect_affordable_upgrades,
    extract_hover_tooltip_crop,
    find_empty_drag_origin,
    find_start_anchor,
    is_same_candidate,
)

# Real screenshot fixtures in parent workspace folder (skipped cleanly if absent)
SCREENSHOT_DIR = Path(__file__).resolve().parent.parent.parent
SHOT_MID_ZOOM = SCREENSHOT_DIR / "RobloxScreenShot20261123_094905285.png"
SHOT_ZOOMED_IN = SCREENSHOT_DIR / "RobloxScreenShot20261123_094851156.png"
SHOT_ZOOMED_OUT = SCREENSHOT_DIR / "RobloxScreenShot20261123_094856406.png"

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
    assert res.start_anchor is not None
    assert res.start_anchor.height_px >= 85


@pytest.mark.skipif(
    not (SHOT_MID_ZOOM.is_file() and SHOT_ZOOMED_IN.is_file() and SHOT_ZOOMED_OUT.is_file()),
    reason="Upgrade tree screenshot fixtures not present",
)
def test_start_anchor_detected_across_all_three_zoom_levels() -> None:
    for shot_path, expected_range in (
        (SHOT_ZOOMED_OUT, (16, 26)),
        (SHOT_MID_ZOOM, (30, 44)),
        (SHOT_ZOOMED_IN, (80, 110)),
    ):
        raw = cv2.imread(str(shot_path))
        assert raw is not None
        img = cv2.resize(raw, (1264, 681), interpolation=cv2.INTER_AREA)
        anchor = find_start_anchor(img)
        assert anchor is not None, f"Start anchor missing on {shot_path.name}"
        assert expected_range[0] <= anchor.height_px <= expected_range[1]
        assert abs(anchor.center[0] - 631) <= 15
        assert abs(anchor.center[1] - 340) <= 15


def test_blank_void_noise_rejected_and_empty_drag_origin_avoids_tree_and_hud() -> None:
    # Faint low-contrast blurred background blobs without Start anchor must NOT be labeled main tree
    noisy_bg = np.full((720, 1280, 3), 24, dtype=np.uint8)
    cv2.circle(noisy_bg, (640, 360), 35, (65, 78, 82), -1)  # dim, low-saturation blurred blob
    scan = detect_affordable_upgrades(noisy_bg)
    assert scan.start_anchor is None
    assert scan.main_tree_bbox is None
    assert scan.tree_coverage_ratio == 0.0

    # Safe empty drag origin must start outside main_tree_bbox + HUD mask
    hud = build_hud_mask(720, 1280)
    main_bbox = (330, 50, 620, 590)
    pair = find_empty_drag_origin(noisy_bg.shape, main_bbox, hud, 0, 140)
    assert pair is not None
    (sx, sy), (ex, ey) = pair
    assert hud[sy, sx] > 0
    assert hud[ey, ex] > 0
    # Outside padded main_tree_bbox
    assert sx < 330 - 16 or sx > 330 + 620 + 16


def test_tooltip_crop_extraction_and_board_denylist_fail_closed() -> None:
    before = np.full((720, 1280, 3), 24, dtype=np.uint8)
    _draw_synthetic_tile(before, 520, 340, "affordable", "$250.0M")
    after = before.copy()
    # Simulate dark slate tooltip card appearing next to hovered tile
    cv2.rectangle(after, (560, 315), (770, 395), (40, 32, 25), -1)
    cv2.putText(after, "Board V", (575, 342), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(after, "+1 Board Size", (575, 368), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    # Also simulate unrelated HUD cash text change in masked bottom-left corner
    cv2.rectangle(after, (20, 520), (180, 560), (20, 200, 20), -1)

    crop = extract_hover_tooltip_crop(before, after, (520, 340))
    assert crop is not None
    assert 190 <= crop.shape[1] <= 230
    assert 65 <= crop.shape[0] <= 95

    assert classify_upgrade_tooltip(["Board V", "+1 Board Size"]).status == "blocked"
    assert classify_upgrade_tooltip(["Board VI", "17 x 17"]).status == "blocked"
    assert classify_upgrade_tooltip(["Boar0 V", "+1 Board"]).status == "blocked"
    assert classify_upgrade_tooltip(["16 x 16 grid"]).status == "blocked"
    assert classify_upgrade_tooltip(["Luck VI", "+5% Luck", "Unlocked"]).status == "unlocked"
    assert classify_upgrade_tooltip([]).status == "unreadable"
    assert classify_upgrade_tooltip(["   "]).status == "unreadable"

    safe = classify_upgrade_tooltip(["Damage XII", "+25% Tower Damage", "$125.0M"])
    assert safe.status == "safe"
    assert safe.title == "Damage XII"
