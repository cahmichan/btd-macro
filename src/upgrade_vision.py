"""Stateless OpenCV vision detection for the BTD hex upgrade tree."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np


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


def segment_tree_clusters(
    frame: Optional[np.ndarray],
    hud_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, int, tuple[float, float], Optional[tuple[int, int, int, int]], set[str]]:
    """Segment hex tile clusters and identify the main middle tree and clipped viewport edges."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return (np.zeros((0, 0), dtype=np.int32), 0, (0.0, 0.0), None, set())

    h, w = frame.shape[:2]
    scale = min(w / 1280.0, h / 720.0)
    mask = hud_mask if hud_mask is not None else build_hud_mask(h, w)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Hex tiles stand out from the dark blurred game background by brightness or color saturation
    tile_mask = (
        (hsv[:, :, 2] >= 75) & ((hsv[:, :, 1] >= 45) | (hsv[:, :, 2] >= 120))
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
    min_area = max(100, int(round(300 * scale * scale)))
    main_id = 0
    best_score = -1.0

    for cid in range(1, num_labels):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < min_area:
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
    """Detect affordable hex tiles via cream-yellow price labels and tree clustering."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return TreeScanResult(candidates=[], all_candidates=[], main_tree_bbox=None, clipped_edges=set())

    h, w = frame.shape[:2]
    scale = min(w / 1280.0, h / 720.0)
    hud_mask = build_hud_mask(h, w)
    labels, main_id, main_centroid, main_bbox, clipped_edges = segment_tree_clusters(frame, hud_mask=hud_mask)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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
) -> np.ndarray:
    """Draw annotated tree bounding box, detected tiles, click points, and edge flags."""
    vis = frame.copy()

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
        f"clipped={edges_text}"
    )
    cv2.rectangle(vis, (8, 8), (min(vis.shape[1] - 8, 680), 34), (15, 15, 15), -1)
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
