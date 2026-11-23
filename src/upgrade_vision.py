"""Stateless OpenCV vision detection for the BTD hex upgrade tree."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import cv2
import numpy as np


@dataclass(frozen=True)
class StartAnchor:
    """Visual anchor for the unique cyan/lime-arrow Start hex tile."""

    center: tuple[int, int]
    bbox: tuple[int, int, int, int]  # (x, y, width, height)
    height_px: int


@dataclass(frozen=True)
class TooltipClassification:
    """Safety verdict from OCR lines of a hovered upgrade tooltip card."""

    status: str  # "safe" | "blocked" | "unlocked" | "unreadable"
    title: str
    lines: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class UpgradeCandidate:
    """Single affordable upgrade tile detected in the upgrade tree."""

    price_center: tuple[int, int]
    click_point: tuple[int, int]
    bbox: tuple[int, int, int, int]  # (x, y, width, height) of the price label
    is_main_tree: bool
    distance_from_center: float
    raw_pixels: int


@dataclass(frozen=True)
class TreeScanResult:
    """Result of scanning an upgrade tree frame for affordable tiles and clipping."""

    candidates: list[UpgradeCandidate]
    all_candidates: list[UpgradeCandidate]
    main_tree_bbox: Optional[tuple[int, int, int, int]]
    clipped_edges: set[str]
    start_anchor: Optional[StartAnchor] = None
    tree_coverage_ratio: float = 0.0


_BOARD_FUZZY_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bboar[d0]\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*x\s*\d+\b", re.IGNORECASE),
)
_UNLOCKED_REGEX = re.compile(r"\bunlocked\b", re.IGNORECASE)


def build_hud_mask(height: int, width: int) -> np.ndarray:
    """Return a uint8 mask (255=active, 0=ignored HUD chrome) for an upgrade tree frame."""
    if height <= 0 or width <= 0:
        return np.zeros((0, 0), dtype=np.uint8)

    mask = np.full((height, width), 255, dtype=np.uint8)
    # Top-left Roblox menu / chat icons
    mask[: int(height * 0.075), : int(width * 0.16)] = 0
    # Top-center "Spend cash to unlock upgrades / Drag to move" header
    mask[: int(height * 0.070), int(width * 0.36) : int(width * 0.64)] = 0
    # Bottom-center red Close button
    mask[int(height * 0.90) :, int(width * 0.38) : int(width * 0.62)] = 0
    # Bottom-left CASH & stats panel
    mask[int(height * 0.51) :, : int(width * 0.19)] = 0
    return mask


def find_start_anchor(
    frame: Optional[np.ndarray],
    *,
    hsv: Optional[np.ndarray] = None,
    hud_mask: Optional[np.ndarray] = None,
) -> Optional[StartAnchor]:
    """Detect the unique cyan Start hex containing the bright lime-green upward arrow."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return None

    h, w = frame.shape[:2]
    scale = min(w / 1280.0, h / 720.0)
    mask = hud_mask if hud_mask is not None else build_hud_mask(h, w)
    hsv_img = hsv if hsv is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_ch, s_ch, v_ch = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    cyan_mask = ((h_ch >= 85) & (h_ch <= 102) & (s_ch >= 140) & (v_ch >= 140)).astype(np.uint8) * 255
    lime_mask = ((h_ch >= 42) & (h_ch <= 72) & (s_ch >= 130) & (v_ch >= 150)).astype(np.uint8) * 255

    cyan_mask = cv2.bitwise_and(cyan_mask, mask)
    lime_mask = cv2.bitwise_and(lime_mask, mask)

    k_sz = max(3, int(round(5 * scale)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
    cyan_closed = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(cyan_closed, connectivity=8)

    min_w = max(8, int(round(10 * scale)))
    min_h = max(10, int(round(12 * scale)))

    best_score = -1
    best_anchor: Optional[StartAnchor] = None
    for cid in range(1, num_labels):
        bx, by, bw, bh, _ = [int(v) for v in stats[cid]]
        if bw < min_w or bh < min_h:
            continue
        aspect = bw / max(1, bh)
        if not (0.55 <= aspect <= 1.35):
            continue

        lime_inside = int(np.count_nonzero(lime_mask[by : by + bh, bx : bx + bw]))
        if lime_inside < max(4, int(0.03 * bw * bh)):
            continue

        if lime_inside > best_score:
            best_score = lime_inside
            cx = max(0, min(w - 1, int(round(float(centroids[cid][0])))))
            cy = max(0, min(h - 1, int(round(float(centroids[cid][1])))))
            best_anchor = StartAnchor(center=(cx, cy), bbox=(bx, by, bw, bh), height_px=bh)

    return best_anchor


def segment_tree_clusters(
    frame: Optional[np.ndarray],
    hud_mask: Optional[np.ndarray] = None,
    *,
    hsv: Optional[np.ndarray] = None,
    start_anchor: Optional[StartAnchor] = None,
) -> tuple[np.ndarray, int, tuple[float, float], Optional[tuple[int, int, int, int]], set[str]]:
    """Segment hex tile clusters and identify the main middle tree and clipped viewport edges."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return (np.zeros((0, 0), dtype=np.int32), 0, (0.0, 0.0), None, set())

    h, w = frame.shape[:2]
    scale = min(w / 1280.0, h / 720.0)
    mask = hud_mask if hud_mask is not None else build_hud_mask(h, w)
    hsv_img = hsv if hsv is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Hex tiles stand out from the dark blurred game background by brightness or color saturation
    tile_mask = (
        (hsv_img[:, :, 2] >= 75) & ((hsv_img[:, :, 1] >= 45) | (hsv_img[:, :, 2] >= 120))
    ).astype(np.uint8) * 255
    tile_mask = cv2.bitwise_and(tile_mask, mask)

    k_close_sz = max(7, int(round(19 * scale)) | 1)
    k_open_sz = max(3, int(round(5 * scale)) | 1)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close_sz, k_close_sz))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open_sz, k_open_sz))

    closed = cv2.morphologyEx(tile_mask, cv2.MORPH_CLOSE, k_close)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

    fcx, fcy = w / 2.0, h / 2.0
    min_area_with_start = max(100, int(round(300 * scale * scale)))
    min_area_no_start = max(1200, int(round(2200 * scale * scale)))
    vivid_mask = (hsv_img[:, :, 1] >= 65) | (hsv_img[:, :, 2] >= 140)

    main_id = 0
    if start_anchor is not None:
        ax, ay = start_anchor.center
        if 0 <= ax < w and 0 <= ay < h:
            cand_id = int(labels[ay, ax])
            if cand_id > 0 and int(stats[cand_id, cv2.CC_STAT_AREA]) >= min_area_with_start:
                main_id = cand_id
            else:
                # Check the Start bounding box in case its center pixel is slightly off a label mask
                bx, by, bw, bh = start_anchor.bbox
                patch = labels[max(0, by) : min(h, by + bh), max(0, bx) : min(w, bx + bw)]
                nz = patch[patch > 0]
                if nz.size > 0:
                    vals, counts = np.unique(nz, return_counts=True)
                    top_id = int(vals[int(np.argmax(counts))])
                    if int(stats[top_id, cv2.CC_STAT_AREA]) >= min_area_with_start:
                        main_id = top_id

    if main_id == 0:
        best_score = -1.0
        min_area = min_area_with_start if start_anchor is not None else min_area_no_start
        for cid in range(1, num_labels):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            if start_anchor is None:
                bx, by, bw, bh, _ = [int(v) for v in stats[cid]]
                comp_mask = labels[by : by + bh, bx : bx + bw] == cid
                vivid_ratio = float(np.mean(vivid_mask[by : by + bh, bx : bx + bw][comp_mask]))
                if vivid_ratio < 0.12:
                    continue
            cx, cy = float(centroids[cid][0]), float(centroids[cid][1])
            dist = math.hypot(cx - fcx, cy - fcy)
            score = area / (1.0 + dist / max(w, h))
            if score > best_score:
                best_score = score
                main_id = cid

    if main_id == 0:
        return (labels, 0, (fcx, fcy), None, set())

    mx, my, mw, mh, _ = [int(v) for v in stats[main_id]]
    main_bbox = (mx, my, mw, mh)
    main_centroid = (float(centroids[main_id][0]), float(centroids[main_id][1]))

    clipped_edges: set[str] = set()
    margin = max(6, int(round(12 * scale)))
    if mx <= margin:
        clipped_edges.add("left")
    if mx + mw >= w - margin:
        clipped_edges.add("right")
    if my <= margin:
        clipped_edges.add("top")
    if my + mh >= h - margin:
        clipped_edges.add("bottom")

    return (labels, main_id, main_centroid, main_bbox, clipped_edges)


def detect_affordable_upgrades(
    frame: Optional[np.ndarray],
    scope: str = "main_only",
    order: str = "center_first",
    tile_click_y_offset_px: int = -14,
) -> TreeScanResult:
    """Detect affordable hex tiles via cream-yellow price labels, Start anchor, and clustering."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return TreeScanResult(
            candidates=[],
            all_candidates=[],
            main_tree_bbox=None,
            clipped_edges=set(),
            start_anchor=None,
            tree_coverage_ratio=0.0,
        )

    h, w = frame.shape[:2]
    scale = min(w / 1280.0, h / 720.0)
    hud_mask = build_hud_mask(h, w)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    start_anchor = find_start_anchor(frame, hsv=hsv, hud_mask=hud_mask)
    labels, main_id, main_centroid, main_bbox, clipped_edges = segment_tree_clusters(
        frame,
        hud_mask=hud_mask,
        hsv=hsv,
        start_anchor=start_anchor,
    )

    if main_id > 0:
        active_px = max(1, int(np.count_nonzero(hud_mask)))
        main_px = int(np.count_nonzero(labels == main_id))
        tree_coverage_ratio = float(main_px / active_px)
    else:
        tree_coverage_ratio = 0.0

    b = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    r = frame[:, :, 2].astype(np.int16)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Warm cream-yellow price text: RGB ≈ (249, 220, 127), HSV ≈ (23, 125, 249)
    price_mask = (
        (h_ch >= 17)
        & (h_ch <= 27)
        & (s_ch >= 85)
        & (s_ch <= 165)
        & (v_ch >= 150)
        & (r >= 155)
        & (g >= 135)
        & (b >= 65)
        & (b <= 175)
        & ((r - b) >= 48)
        & ((r - g) >= 6)
        & ((r - g) <= 50)
    ).astype(np.uint8) * 255
    price_mask = cv2.bitwise_and(price_mask, hud_mask)

    # Dilate horizontally only to join price characters without merging vertical neighbors
    kw = max(5, int(round(7 * scale)) | 1)
    dilated = cv2.dilate(price_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1)))
    pnum, _, pstats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    min_raw_px = max(12, int(round(18 * scale * scale)))
    min_h = max(3, int(round(3 * scale)))
    max_h = max(14, int(round(18 * scale)))
    min_w = max(11, int(round(12 * scale)))
    max_w = max(65, int(round(85 * scale)))
    min_amber_px = max(12, int(round(20 * scale * scale)))

    all_candidates: list[UpgradeCandidate] = []
    for idx in range(1, pnum):
        px, py, pw, ph, _ = [int(v) for v in pstats[idx]]
        sub = price_mask[py : py + ph, px : px + pw]
        ys, xs = np.where(sub > 0)
        raw_px = int(ys.size)
        if raw_px < min_raw_px:
            continue

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        raw_w = x_max - x_min + 1
        raw_h = y_max - y_min + 1
        raw_fill = raw_px / max(1, raw_w * raw_h)
        raw_aspect = raw_w / max(1, raw_h)

        if not (min_h <= raw_h <= max_h and min_w <= raw_w <= max_w and raw_aspect >= 2.2 and raw_fill >= 0.30):
            continue

        cx = max(0, min(w - 1, px + int(round(float(xs.mean())))))
        cy = max(0, min(h - 1, py + int(round(float(ys.mean())))))

        # 1. Confirm dimmed/dark interior immediately around price (filters unlocked bright gold Cash tiles)
        sy1 = max(0, py - max(4, int(round(8 * scale))))
        sy2 = min(h, py + ph + max(2, int(round(4 * scale))))
        sx1 = max(0, px - 2)
        sx2 = min(w, px + pw + 2)
        strip = hsv[sy1:sy2, sx1:sx2]
        if strip.size == 0 or float(np.mean(strip[:, :, 2] < 115)) < 0.35:
            continue

        # 2. Confirm a solid hex body directly above price (filters floating legend text like FLAGS)
        by1 = max(0, cy - int(round(22 * scale)))
        by2 = min(h, cy - int(round(5 * scale)))
        bx1 = max(0, cx - int(round(12 * scale)))
        bx2 = min(w, cx + int(round(12 * scale)))
        if by2 <= by1 or bx2 <= bx1:
            continue
        body = hsv[by1:by2, bx1:bx2]
        if body.size == 0 or float(np.mean(body[:, :, 2] >= 40)) < 0.80:
            continue

        # 3. Confirm amber/gold unpurchased hex border in the local neighborhood
        ry1 = max(0, cy - int(round(30 * scale)))
        ry2 = min(h, cy + int(round(12 * scale)))
        rx1 = max(0, cx - int(round(28 * scale)))
        rx2 = min(w, cx + int(round(28 * scale)))
        ring = hsv[ry1:ry2, rx1:rx2]
        amber_px = int(
            np.count_nonzero(
                (ring[:, :, 0] >= 10)
                & (ring[:, :, 0] <= 32)
                & (ring[:, :, 1] >= 80)
                & (ring[:, :, 2] >= 100)
            )
        )
        if amber_px < min_amber_px:
            continue

        click_y = max(0, min(h - 1, cy + int(round(tile_click_y_offset_px * scale))))
        click_point = (cx, click_y)
        is_main = bool(main_id > 0 and (labels[cy, cx] == main_id or labels[click_y, cx] == main_id))
        dist = math.hypot(cx - main_centroid[0], cy - main_centroid[1])

        all_candidates.append(
            UpgradeCandidate(
                price_center=(cx, cy),
                click_point=click_point,
                bbox=(px + x_min, py + y_min, raw_w, raw_h),
                is_main_tree=is_main,
                distance_from_center=dist,
                raw_pixels=raw_px,
            )
        )

    if order == "top_down":
        all_candidates.sort(
            key=lambda c: (0 if c.is_main_tree else 1, c.price_center[1], c.price_center[0])
        )
    else:
        all_candidates.sort(
            key=lambda c: (
                0 if c.is_main_tree else 1,
                c.distance_from_center,
                c.price_center[1],
                c.price_center[0],
            )
        )

    if scope == "main_only":
        filtered = [c for c in all_candidates if c.is_main_tree]
    else:
        filtered = list(all_candidates)

    return TreeScanResult(
        candidates=filtered,
        all_candidates=all_candidates,
        main_tree_bbox=main_bbox,
        clipped_edges=clipped_edges,
        start_anchor=start_anchor,
        tree_coverage_ratio=tree_coverage_ratio,
    )


def find_empty_drag_origin(
    frame_shape: tuple[int, ...],
    main_tree_bbox: Optional[tuple[int, int, int, int]],
    hud_mask: Optional[np.ndarray],
    drag_dx: int,
    drag_dy: int,
) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    """Find a drag segment ((x1, y1), (x2, y2)) starting in safe empty dark background."""
    if len(frame_shape) < 2:
        return None
    h, w = int(frame_shape[0]), int(frame_shape[1])
    if h <= 32 or w <= 32:
        return None

    scale = min(w / 1280.0, h / 720.0)
    mask = hud_mask.copy() if hud_mask is not None else build_hud_mask(h, w)

    border = max(12, int(round(16 * scale)))
    if h > 2 * border and w > 2 * border:
        mask[:border, :] = 0
        mask[h - border :, :] = 0
        mask[:, :border] = 0
        mask[:, w - border :] = 0

    end_valid = (mask > 0).copy()

    if main_tree_bbox is not None:
        mx, my, mw, mh = main_tree_bbox
        pad = max(16, int(round(20 * scale)))
        x1 = max(0, mx - pad)
        y1 = max(0, my - pad)
        x2 = min(w, mx + mw + pad)
        y2 = min(h, my + mh + pad)
        mask[y1:y2, x1:x2] = 0

    if not np.any(mask > 0):
        return None

    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    dx = int(drag_dx)
    dy = int(drag_dy)

    # Shift end_valid back by (-dx, -dy) to constrain valid start pixels (sx, sy)
    shifted_end = np.zeros((h, w), dtype=bool)
    src_y1 = max(0, dy)
    src_y2 = min(h, h + dy)
    src_x1 = max(0, dx)
    src_x2 = min(w, w + dx)
    dst_y1 = max(0, -dy)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x1 = max(0, -dx)
    dst_x2 = dst_x1 + (src_x2 - src_x1)

    if src_y2 <= src_y1 or src_x2 <= src_x1:
        return None
    shifted_end[dst_y1:dst_y2, dst_x1:dst_x2] = end_valid[src_y1:src_y2, src_x1:src_x2]

    candidate_scores = np.where((mask > 0) & shifted_end, dist_map, 0.0)
    best_val = float(candidate_scores.max())
    if best_val <= 0.0:
        return None

    by, bx = np.unravel_index(int(np.argmax(candidate_scores)), candidate_scores.shape)
    sx, sy = int(bx), int(by)
    ex, ey = sx + dx, sy + dy
    return ((sx, sy), (ex, ey))


def extract_hover_tooltip_crop(
    before_frame: Optional[np.ndarray],
    after_frame: Optional[np.ndarray],
    hover_point: tuple[int, int],
) -> Optional[np.ndarray]:
    """Isolate the newly appeared dark hover tooltip card via frame diff near hover_point."""
    if (
        before_frame is None
        or after_frame is None
        or before_frame.size == 0
        or after_frame.size == 0
        or before_frame.shape != after_frame.shape
        or after_frame.ndim != 3
    ):
        return None

    h, w = after_frame.shape[:2]
    scale = min(w / 1280.0, h / 720.0)
    hud_mask = build_hud_mask(h, w)

    diff = cv2.absdiff(before_frame, after_frame)
    diff_max = np.max(diff, axis=2)
    changed = ((diff_max >= 14).astype(np.uint8) * 255)
    changed = cv2.bitwise_and(changed, hud_mask)

    k_close_sz = max(7, int(round(11 * scale)) | 1)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close_sz, k_close_sz))
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, k_close)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    hx, hy = int(hover_point[0]), int(hover_point[1])

    min_w = max(50, int(round(65 * scale)))
    max_w = max(200, int(round(450 * scale)))
    min_h = max(18, int(round(24 * scale)))
    max_h = max(100, int(round(240 * scale)))
    max_dist = max(180.0, 320.0 * scale)

    hsv_after = cv2.cvtColor(after_frame, cv2.COLOR_BGR2HSV)
    best_dist = float("inf")
    best_bbox: Optional[tuple[int, int, int, int]] = None

    for cid in range(1, num_labels):
        bx, by, bw, bh, area = [int(v) for v in stats[cid]]
        if not (min_w <= bw <= max_w and min_h <= bh <= max_h):
            continue
        fill = area / max(1, bw * bh)
        if fill < 0.45:
            continue

        # Point-to-rectangle distance from hover_point
        dx = max(bx - hx, 0, hx - (bx + bw))
        dy = max(by - hy, 0, hy - (by + bh))
        dist = math.hypot(dx, dy)
        if dist > max_dist:
            continue

        patch_v = hsv_after[by : by + bh, bx : bx + bw, 2]
        if patch_v.size == 0 or float(np.mean(patch_v <= 80)) < 0.25:
            continue

        if dist < best_dist:
            best_dist = dist
            best_bbox = (bx, by, bw, bh)

    if best_bbox is None:
        return None

    bx, by, bw, bh = best_bbox
    pad = max(2, int(round(3 * scale)))
    x1 = max(0, bx - pad)
    y1 = max(0, by - pad)
    x2 = min(w, bx + bw + pad)
    y2 = min(h, by + bh + pad)
    return np.ascontiguousarray(after_frame[y1:y2, x1:x2], dtype=np.uint8)


def classify_upgrade_tooltip(
    lines: Sequence[str],
    blocked_keywords: Sequence[str] = ("board",),
) -> TooltipClassification:
    """Classify OCR tooltip lines into safe, blocked, unlocked, or unreadable (fail-closed)."""
    cleaned = [str(line).strip() for line in lines if str(line).strip()]
    if not cleaned:
        return TooltipClassification(
            status="unreadable",
            title="",
            lines=(),
            reason="empty_ocr",
        )

    title = cleaned[0]
    joined = " ".join(cleaned)
    joined_lower = joined.lower()

    if _UNLOCKED_REGEX.search(joined):
        return TooltipClassification(
            status="unlocked",
            title=title,
            lines=tuple(cleaned),
            reason="unlocked",
        )

    for kw in blocked_keywords:
        token = str(kw).strip().lower()
        if token and token in joined_lower:
            return TooltipClassification(
                status="blocked",
                title=title,
                lines=tuple(cleaned),
                reason=f"keyword:{token}",
            )

    for pattern in _BOARD_FUZZY_REGEXES:
        if pattern.search(joined):
            return TooltipClassification(
                status="blocked",
                title=title,
                lines=tuple(cleaned),
                reason=f"board_regex:{pattern.pattern}",
            )

    return TooltipClassification(
        status="safe",
        title=title,
        lines=tuple(cleaned),
        reason="ok",
    )


def is_same_candidate(
    a: Union[UpgradeCandidate, tuple[int, int]],
    b: Union[UpgradeCandidate, tuple[int, int]],
    tolerance_px: int = 24,
) -> bool:
    """Return True if two upgrade candidates or click points lie within tolerance_px."""
    ax, ay = a.click_point if isinstance(a, UpgradeCandidate) else (int(a[0]), int(a[1]))
    bx, by = b.click_point if isinstance(b, UpgradeCandidate) else (int(b[0]), int(b[1]))
    return math.hypot(ax - bx, ay - by) <= float(tolerance_px)


def render_upgrade_debug_overlay(
    frame: np.ndarray,
    result: TreeScanResult,
    scope: str = "main_only",
    min_start_height_px: int = 30,
    max_start_height_px: int = 44,
) -> np.ndarray:
    """Draw annotated tree bounding box, Start anchor, detected tiles, and edge/zoom status."""
    vis = frame.copy()
    h, w = vis.shape[:2]
    scale = min(w / 1280.0, h / 720.0)

    if result.main_tree_bbox is not None:
        mx, my, mw, mh = result.main_tree_bbox
        cv2.rectangle(vis, (mx, my), (mx + mw, my + mh), (255, 255, 0), 2)
        cv2.putText(
            vis,
            "MAIN TREE",
            (mx + 6, max(20, my + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    zoom_status = "NO_START"
    start_h_label = "none"
    if result.start_anchor is not None:
        sa = result.start_anchor
        sx, sy, sw, sh = sa.bbox
        cv2.rectangle(vis, (sx, sy), (sx + sw, sy + sh), (255, 200, 0), 2)
        cv2.drawMarker(
            vis,
            sa.center,
            (0, 255, 128),
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
        )
        cv2.putText(
            vis,
            f"START h={sa.height_px}px",
            (sx, max(18, sy - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 220, 0),
            1,
            cv2.LINE_AA,
        )
        min_scaled = max(10, int(round(min_start_height_px * scale)))
        max_scaled = max(min_scaled + 4, int(round(max_start_height_px * scale)))
        start_h_label = f"{sa.height_px}px"
        if sa.height_px < min_scaled:
            zoom_status = "TOO_SMALL"
        elif sa.height_px > max_scaled:
            zoom_status = "TOO_BIG"
        else:
            zoom_status = "OK"

    active_ranks = {c.click_point: idx + 1 for idx, c in enumerate(result.candidates)}

    for cand in result.all_candidates:
        x, y, bw, bh = cand.bbox
        rank = active_ranks.get(cand.click_point)
        if rank is not None:
            color = (0, 230, 0)
            tag = f"#{rank} {'M' if cand.is_main_tree else 'S'}"
        else:
            color = (0, 165, 255)
            tag = "SIDE (skipped)"

        cv2.rectangle(vis, (x - 2, y - 2), (x + bw + 2, y + bh + 2), color, 2)
        cv2.line(vis, cand.price_center, cand.click_point, color, 1, cv2.LINE_AA)
        cv2.circle(vis, cand.click_point, 5, color, -1)
        cv2.circle(vis, cand.click_point, 7, (0, 0, 0), 1)
        cv2.putText(
            vis,
            tag,
            (cand.click_point[0] + 8, max(16, cand.click_point[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    edges_text = ",".join(sorted(result.clipped_edges)) if result.clipped_edges else "none"
    summary = (
        f"scope={scope} | active={len(result.candidates)} | total={len(result.all_candidates)} | "
        f"start_h={start_h_label} ({zoom_status}) | cov={result.tree_coverage_ratio:.2f} | clipped={edges_text}"
    )
    cv2.rectangle(vis, (8, 8), (min(vis.shape[1] - 8, 980), 34), (15, 15, 15), -1)
    cv2.putText(
        vis,
        summary,
        (14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis
