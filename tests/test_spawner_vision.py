"""Unit and fixture tests for Spawner Shop green Buy pill detection and safety helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.spawner_vision import (
    SpawnerBuyButton,
    detect_spawner_buy_buttons,
    did_spawner_purchase_register,
    extract_buy_pill_crop,
    is_buy_pill_text_valid,
    is_scroll_grid_unchanged,
)

SCREENSHOT_DIR = Path(__file__).resolve().parent.parent.parent
SHOT_SPAWNERS_7 = SCREENSHOT_DIR / "RobloxScreenShot20261123_155501482.png"
SHOT_SPAWNERS_SOLD_OUT = SCREENSHOT_DIR / "RobloxScreenShot20261123_155330225.png"


@pytest.mark.skipif(not SHOT_SPAWNERS_7.is_file(), reason="Spawner 7-stock screenshot fixture not present")
@pytest.mark.parametrize("size", [None, (1280, 720), (1264, 681)])
def test_spawner_fixture_detects_exact_7_buy_buttons_cheapest_first(
    size: tuple[int, int] | None,
) -> None:
    raw = cv2.imread(str(SHOT_SPAWNERS_7))
    assert raw is not None
    frame = raw if size is None else cv2.resize(raw, size, interpolation=cv2.INTER_AREA)

    res = detect_spawner_buy_buttons(frame)
    assert len(res.buttons) == 7

    # Row 0 has 4 cards (Forest, Barracks, Bandit Camp, Goblin Camp)
    row0 = [b for b in res.buttons if b.row_index == 0]
    row1 = [b for b in res.buttons if b.row_index == 1]
    assert len(row0) == 4
    assert len(row1) == 3

    # Cheapest-first: left-to-right within each row, top row before bottom row
    assert [b.center[0] for b in row0] == sorted(b.center[0] for b in row0)
    assert [b.center[0] for b in row1] == sorted(b.center[0] for b in row1)
    assert max(b.center[1] for b in row0) < min(b.center[1] for b in row1)

    # Every click center lies strictly inside the green pill bbox and scroll grid ROI
    rx1, ry1, rx2, ry2 = res.grid_roi_px
    for btn in res.buttons:
        bx, by, bw, bh = btn.bbox
        cx, cy = btn.center
        assert rx1 <= cx <= rx2 and ry1 <= cy <= ry2
        assert bx < cx < bx + bw
        assert by < cy < by + bh
        crop = extract_buy_pill_crop(frame, btn)
        assert crop is not None and crop.size > 0


@pytest.mark.skipif(not SHOT_SPAWNERS_SOLD_OUT.is_file(), reason="Spawner all-sold-out screenshot fixture not present")
@pytest.mark.parametrize("size", [None, (1280, 720), (1264, 681)])
def test_spawner_fixture_detects_0_buttons_when_all_sold_out(
    size: tuple[int, int] | None,
) -> None:
    raw = cv2.imread(str(SHOT_SPAWNERS_SOLD_OUT))
    assert raw is not None
    frame = raw if size is None else cv2.resize(raw, size, interpolation=cv2.INTER_AREA)

    res = detect_spawner_buy_buttons(frame)
    assert res.buttons == []


def test_spawner_robux_and_outside_roi_never_detected() -> None:
    frame = np.full((720, 1280, 3), 30, dtype=np.uint8)
    green_bgr = (70, 200, 60)  # vibrant green HSV ~ (62, 179, 200)
    navy_robux_bgr = (90, 50, 30)

    # Valid green pill inside ROI
    cv2.rectangle(frame, (395, 340), (497, 368), green_bgr, -1)
    # Dark navy Robux pill directly below it
    cv2.rectangle(frame, (395, 374), (497, 400), navy_robux_bgr, -1)
    # Green button outside ROI (top-left header / bottom-left cash panel)
    cv2.rectangle(frame, (40, 620), (142, 648), green_bgr, -1)
    cv2.rectangle(frame, (400, 40), (502, 68), green_bgr, -1)

    res = detect_spawner_buy_buttons(frame)
    assert len(res.buttons) == 1
    assert res.buttons[0].center == (446, 354)


def test_buy_pill_text_validation_purchase_registration_and_scroll_end() -> None:
    assert is_buy_pill_text_valid(["Buy 1,000"]) is True
    assert is_buy_pill_text_valid(["Buy 30.0K"]) is True
    assert is_buy_pill_text_valid(["Buya,800"]) is True  # OCR glued "Buy 1,800"
    assert is_buy_pill_text_valid(["Buya,800 Sold out"]) is False
    assert is_buy_pill_text_valid(["Sold out"]) is False
    assert is_buy_pill_text_valid(["30"]) is False
    assert is_buy_pill_text_valid(["Buy 30 Robux"]) is False
    assert is_buy_pill_text_valid([]) is False

    btn = SpawnerBuyButton(center=(446, 354), bbox=(395, 340, 102, 28), row_index=0, col_index=0, fill_ratio=0.9)
    prev = np.full((720, 1280, 3), 40, dtype=np.uint8)
    same = prev.copy()
    # Animated battle noise outside the inner scroll grid ROI should be ignored
    same[50:150, 20:200] = 220
    assert is_scroll_grid_unchanged(prev, same) is True
    assert did_spawner_purchase_register(prev, same, btn) is False

    # Updating the card's stock/owned patch directly above the pill registers purchase even if cash header didn't change
    stock_changed = prev.copy()
    cv2.putText(stock_changed, "x2 in stock", (400, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    assert did_spawner_purchase_register(prev, stock_changed, btn) is True

    # Scrolling changes content inside the inner scroll grid ROI
    scrolled = prev.copy()
    scrolled[220:550, 400:800] = 180
    assert is_scroll_grid_unchanged(prev, scrolled) is False
