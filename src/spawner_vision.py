"""Stateless OpenCV vision detection for the BTD Spawner Shop modal."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

SPAWNER_GRID_ROI: tuple[float, float, float, float] = (0.29, 0.20, 0.70, 0.93)
SPAWNER_SCROLL_COMPARE_ROI: tuple[float, float, float, float] = (0.29, 0.20, 0.70, 0.92)
SPAWNER_CASH_HEADER_ROI: tuple[float, float, float, float] = (0.42, 0.16, 0.58, 0.20)

# No word boundary after "buy": Windows OCR often glues the price on (e.g. "Buy 1,800" -> "Buya,800").
# Sold out / Robux / Restock pills are still rejected by _REJECT_WORDS_RE.
_BUY_START_RE = re.compile(r"^\s*buy", re.IGNORECASE)
_REJECT_WORDS_RE = re.compile(r"\b(sold|out|robux|restock)\b", re.IGNORECASE)


@dataclass(frozen=True)
class SpawnerBuyButton:
    """In-stock green cash Buy pill inside the Spawner Shop scroll grid."""

    center: tuple[int, int]
    bbox: tuple[int, int, int, int]  # (x, y, width, height) in client pixels
    row_index: int
    col_index: int
    fill_ratio: float


@dataclass(frozen=True)
class SpawnerScanResult:
    """Detected green Buy pills ordered cheapest-first inside the scroll grid ROI."""

    buttons: list[SpawnerBuyButton]
    grid_roi_px: tuple[int, int, int, int]  # (x1, y1, x2, y2)


def _norm_roi_to_px(
    width: int,
    height: int,
    roi: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    x1 = max(0, min(width, int(math.floor(roi[0] * width))))
    y1 = max(0, min(height, int(math.floor(roi[1] * height))))
    x2 = max(0, min(width, int(math.ceil(roi[2] * width))))
    y2 = max(0, min(height, int(math.ceil(roi[3] * height))))
    return (x1, y1, x2, y2)


def detect_spawner_buy_buttons(frame: Optional[np.ndarray]) -> SpawnerScanResult:
    """Detect green cash Buy buttons strictly inside the Spawner Shop scroll grid."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return SpawnerScanResult(buttons=[], grid_roi_px=(0, 0, 0, 0))

    h, w = frame.shape[:2]
    scale = min(w / 1280.0, h / 720.0)
    rx1, ry1, rx2, ry2 = _norm_roi_to_px(w, h, SPAWNER_GRID_ROI)
    if rx2 <= rx1 or ry2 <= ry1:
        return SpawnerScanResult(buttons=[], grid_roi_px=(rx1, ry1, rx2, ry2))

    roi_frame = frame[ry1:ry2, rx1:rx2]
    roi_h, roi_w = roi_frame.shape[:2]
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    green_mask = (
        (hsv[:, :, 0] >= 50)
        & (hsv[:, :, 0] <= 82)
        & (hsv[:, :, 1] >= 110)
        & (hsv[:, :, 2] >= 120)
    ).astype(np.uint8) * 255

    k_sz = max(3, int(round(5 * scale)) | 1)
    closed = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (k_sz, k_sz)),
    )

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    raw_pills: list[tuple[int, int, int, int, float]] = []
    for cid in range(1, num_labels):
        bx, by, bw, bh, area = [int(v) for v in stats[cid]]
        w_scaled = bw / max(1e-6, scale)
        h_scaled = bh / max(1e-6, scale)
        aspect = bw / max(1, bh)
        fill = area / max(1, bw * bh)
        bottom_margin = roi_h - (by + bh)
        top_margin = by

        if (
            65.0 <= w_scaled <= 135.0
            and 20.0 <= h_scaled <= 38.0
            and 3.0 <= aspect <= 4.8
            and fill >= 0.70
            and bottom_margin >= 4
            and top_margin >= 2
        ):
            raw_pills.append((rx1 + bx, ry1 + by, bw, bh, float(fill)))

    row_bucket = max(1, int(round(40 * scale)))
    raw_pills.sort(key=lambda p: (p[1] // row_bucket, p[0]))

    buttons: list[SpawnerBuyButton] = []
    if raw_pills:
        # Assign clean row_index (0, 1, ...) and col_index within each visual row
        current_row_y: Optional[int] = None
        row_idx = -1
        col_idx = 0
        row_tolerance = max(12, int(round(36 * scale)))

        for gx, gy, bw, bh, fill in raw_pills:
            if current_row_y is None or abs(gy - current_row_y) > row_tolerance:
                row_idx += 1
                col_idx = 0
                current_row_y = gy
            else:
                col_idx += 1

            cx = gx + bw // 2
            cy = gy + bh // 2
            buttons.append(
                SpawnerBuyButton(
                    center=(cx, cy),
                    bbox=(gx, gy, bw, bh),
                    row_index=row_idx,
                    col_index=col_idx,
                    fill_ratio=fill,
                )
            )

    return SpawnerScanResult(buttons=buttons, grid_roi_px=(rx1, ry1, rx2, ry2))


def extract_buy_pill_crop(
    frame: Optional[np.ndarray],
    button: SpawnerBuyButton,
) -> Optional[np.ndarray]:
    """Return a tight BGR crop of a detected Buy pill for optional OCR verification."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return None
    h, w = frame.shape[:2]
    bx, by, bw, bh = button.bbox
    x1 = max(0, bx + 2)
    y1 = max(0, by + 2)
    x2 = min(w, bx + bw - 2)
    y2 = min(h, by + bh - 2)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.ascontiguousarray(frame[y1:y2, x1:x2], dtype=np.uint8)


def is_buy_pill_text_valid(lines: Sequence[str]) -> bool:
    """Return True if OCR lines on a pill confirm it is a cash Buy button (never Sold out/Robux)."""
    cleaned = [str(line).strip() for line in lines if str(line).strip()]
    if not cleaned:
        return False
    joined = " ".join(cleaned)
    if _REJECT_WORDS_RE.search(joined):
        return False
    return bool(_BUY_START_RE.match(joined))


_STOCK_COUNT_RE = re.compile(r"\bx\s*(\d{1,3})\s*in\b", re.IGNORECASE)
_SOLD_OUT_RE = re.compile(r"\bsold\s*out\b", re.IGNORECASE)


def parse_spawner_stock(lines: Sequence[str]) -> Optional[int]:
    """Parse a card's stock from OCR lines: "x3 in stock" -> 3, "SOLD OUT" -> 0, unreadable -> None."""
    joined = " ".join(str(line).strip() for line in lines if str(line).strip())
    if not joined:
        return None
    if _SOLD_OUT_RE.search(joined):
        return 0
    m = _STOCK_COUNT_RE.search(joined)
    return int(m.group(1)) if m else None


def extract_card_stock_patch(
    frame: Optional[np.ndarray],
    button: SpawnerBuyButton,
) -> Optional[np.ndarray]:
    """Extract the card stock/owned text region immediately above a green Buy pill."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return None
    h, w = frame.shape[:2]
    scale = min(w / 1280.0, h / 720.0)
    bx, by, bw, _ = button.bbox
    y1 = max(0, by - int(round(90 * scale)))
    y2 = max(y1 + 1, by - max(2, int(round(4 * scale))))
    x1 = max(0, bx)
    x2 = min(w, bx + bw)
    if y2 <= y1 or x2 <= x1:
        return None
    return np.ascontiguousarray(frame[y1:y2, x1:x2], dtype=np.uint8)


def extract_cash_header_patch(frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Extract the 'Your cash: ...' strip at the top of the Spawner Shop modal."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _norm_roi_to_px(w, h, SPAWNER_CASH_HEADER_ROI)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.ascontiguousarray(frame[y1:y2, x1:x2], dtype=np.uint8)


def _mean_abs_diff(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None or a.size == 0 or b.size == 0 or a.shape != b.shape:
        return None
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def did_spawner_purchase_register(
    prev_frame: Optional[np.ndarray],
    cur_frame: Optional[np.ndarray],
    button: SpawnerBuyButton,
    diff_threshold: float = 1.5,
    check_cash: bool = True,
) -> bool:
    """Return True if the card's stock/owned patch (or, if check_cash, the cash header) visibly changed."""
    if prev_frame is None or cur_frame is None or prev_frame.shape != cur_frame.shape:
        return False

    stock_prev = extract_card_stock_patch(prev_frame, button)
    stock_cur = extract_card_stock_patch(cur_frame, button)
    stock_diff = _mean_abs_diff(stock_prev, stock_cur)
    if stock_diff is not None and stock_diff >= diff_threshold:
        return True
    if not check_cash:
        return False

    cash_prev = extract_cash_header_patch(prev_frame)
    cash_cur = extract_cash_header_patch(cur_frame)
    cash_diff = _mean_abs_diff(cash_prev, cash_cur)
    return bool(cash_diff is not None and cash_diff >= diff_threshold)


def is_scroll_grid_unchanged(
    prev_frame: Optional[np.ndarray],
    cur_frame: Optional[np.ndarray],
    diff_threshold: float = 1.5,
    corr_threshold: float = 0.992,
) -> bool:
    """Compare only the inner Spawner Shop scroll grid ROI to ignore animated background battles."""
    if (
        prev_frame is None
        or cur_frame is None
        or prev_frame.size == 0
        or cur_frame.size == 0
        or prev_frame.shape != cur_frame.shape
    ):
        return False

    h, w = prev_frame.shape[:2]
    x1, y1, x2, y2 = _norm_roi_to_px(w, h, SPAWNER_SCROLL_COMPARE_ROI)
    if x2 <= x1 or y2 <= y1:
        return False

    p_roi = prev_frame[y1:y2, x1:x2]
    c_roi = cur_frame[y1:y2, x1:x2]
    mad = float(np.mean(np.abs(p_roi.astype(np.float32) - c_roi.astype(np.float32))))
    if mad < diff_threshold:
        return True

    # If either ROI is virtually uniform, pixel MAD already tells us whether content changed
    if float(np.std(p_roi)) < 2.0 or float(np.std(c_roi)) < 2.0:
        return False

    res = cv2.matchTemplate(c_roi, p_roi, cv2.TM_CCOEFF_NORMED)
    corr = float(res[0, 0]) if res.size > 0 else 0.0
    return math.isfinite(corr) and corr > corr_threshold


def render_spawner_debug_overlay(
    frame: np.ndarray,
    result: SpawnerScanResult,
) -> np.ndarray:
    """Draw scroll grid ROI, detected green Buy pills, click crosshairs, and cheapest-first order."""
    vis = frame.copy()
    rx1, ry1, rx2, ry2 = result.grid_roi_px
    if rx2 > rx1 and ry2 > ry1:
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

    for idx, btn in enumerate(result.buttons, start=1):
        bx, by, bw, bh = btn.bbox
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 235, 0), 2)
        cv2.drawMarker(
            vis,
            btn.center,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=2,
        )
        cv2.putText(
            vis,
            f"#{idx}",
            (bx + 4, max(16, by - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    summary = f"Spawner Buy pills: {len(result.buttons)} (cheapest-first order)"
    cv2.rectangle(vis, (8, 8), (min(vis.shape[1] - 8, 520), 34), (15, 15, 15), -1)
    cv2.putText(
        vis,
        summary,
        (14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis
