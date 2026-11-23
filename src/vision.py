"""Screen capture, BGR OpenCV template matching, and failure snapshot rotation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import cv2
import mss
import numpy as np

from src.config import (
    get_assets_dir,
    get_dismiss_dir,
    get_failures_dir,
)

REQUIRED_TEMPLATES: tuple[str, ...] = (
    "btn_battle",
    "modal_disconnected",
    "btn_reconnect",
)

OPTIONAL_TEMPLATES: tuple[str, ...] = (
    "hud_in_run",
    "btn_yes_continue",
    "btn_continue",
    "btn_start_over",
    "modal_same_account",
    "btn_upgrades",
    "btn_upgrades_close",
    "btn_spawners",
    "btn_spawners_close",
)


@dataclass(frozen=True)
class MatchResult:
    """Result of a single template matching operation in client coordinates."""

    found: bool
    confidence: float
    center: Optional[tuple[int, int]] = None
    bbox: Optional[tuple[int, int, int, int]] = None  # (x, y, width, height)
    template_name: str = ""


def _read_bgr_image(path: Path) -> Optional[np.ndarray]:
    """Load an image in BGR format safely on Windows paths with spaces or Unicode."""
    if not path.is_file():
        return None
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return None
        return np.ascontiguousarray(img, dtype=np.uint8)
    except Exception:
        return None


def _write_png_image(path: Path, image: np.ndarray) -> bool:
    """Write a BGR image to PNG safely on Windows paths."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


class VisionEngine:
    """Handles client-area screen capture, BGR template matching, and failure snapshots."""

    def __init__(
        self,
        assets_dir: Optional[Path] = None,
        dismiss_dir: Optional[Path] = None,
        failures_dir: Optional[Path] = None,
        max_failure_snapshots: int = 20,
    ) -> None:
        self.assets_dir = Path(assets_dir) if assets_dir is not None else get_assets_dir()
        self.dismiss_dir = Path(dismiss_dir) if dismiss_dir is not None else get_dismiss_dir()
        self.failures_dir = Path(failures_dir) if failures_dir is not None else get_failures_dir()
        self.max_failure_snapshots = max(1, int(max_failure_snapshots))

        self._templates: dict[str, np.ndarray] = {}
        self._dismiss_templates: dict[str, np.ndarray] = {}
        self.reload_templates()

    def reload_templates(self) -> None:
        """Load or reload all main and optional dismiss templates from disk."""
        self._templates.clear()
        self._dismiss_templates.clear()

        for name in REQUIRED_TEMPLATES + OPTIONAL_TEMPLATES:
            img = _read_bgr_image(self.assets_dir / f"{name}.png")
            if img is not None:
                self._templates[name] = img

        if self.dismiss_dir.is_dir():
            for png_path in sorted(self.dismiss_dir.glob("*.png")):
                img = _read_bgr_image(png_path)
                if img is not None:
                    self._dismiss_templates[png_path.stem] = img

    def missing_required_templates(self) -> list[str]:
        """Return names of required templates not currently loaded."""
        return [name for name in REQUIRED_TEMPLATES if name not in self._templates]

    def has_template(self, name: str) -> bool:
        """Return True if the named main template is loaded."""
        return name in self._templates

    def capture_client(self, client_rect: tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Capture only the Roblox client area (left, top, width, height) as a BGR ndarray."""
        left, top, width, height = client_rect
        if width <= 0 or height <= 0:
            return None
        monitor = {
            "left": int(left),
            "top": int(top),
            "width": int(width),
            "height": int(height),
        }
        try:
            with mss.mss() as sct:
                raw = sct.grab(monitor)
                arr = np.asarray(raw, dtype=np.uint8)
                if arr.ndim != 3 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
                    return None
                return np.ascontiguousarray(arr[:, :, :3], dtype=np.uint8)
        except Exception:
            return None

    @staticmethod
    def match_template(
        frame: Optional[np.ndarray],
        template: Optional[np.ndarray],
        threshold: float = 0.85,
        roi: Optional[Sequence[float]] = None,
        template_name: str = "",
    ) -> MatchResult:
        """Run BGR TM_CCOEFF_NORMED matching inside an optional normalized ROI [x1, y1, x2, y2]."""
        if frame is None or template is None or frame.size == 0 or template.size == 0:
            return MatchResult(found=False, confidence=0.0, template_name=template_name)

        fh, fw = frame.shape[:2]
        th, tw = template.shape[:2]

        x1, y1, x2, y2 = 0, 0, fw, fh
        if roi is not None and len(roi) == 4:
            rx1 = max(0.0, min(1.0, float(roi[0])))
            ry1 = max(0.0, min(1.0, float(roi[1])))
            rx2 = max(0.0, min(1.0, float(roi[2])))
            ry2 = max(0.0, min(1.0, float(roi[3])))
            x1 = int(math.floor(rx1 * fw))
            y1 = int(math.floor(ry1 * fh))
            x2 = int(math.ceil(rx2 * fw))
            y2 = int(math.ceil(ry2 * fh))

        if x2 <= x1 or y2 <= y1:
            return MatchResult(found=False, confidence=0.0, template_name=template_name)

        search_region = frame[y1:y2, x1:x2]
        sh, sw = search_region.shape[:2]
        if th > sh or tw > sw:
            return MatchResult(found=False, confidence=0.0, template_name=template_name)

        res = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if not math.isfinite(max_val):
            return MatchResult(found=False, confidence=0.0, template_name=template_name)

        conf = float(max_val)
        match_x = x1 + int(max_loc[0])
        match_y = y1 + int(max_loc[1])
        center = (match_x + tw // 2, match_y + th // 2)
        bbox = (match_x, match_y, tw, th)
        found = conf >= float(threshold)
        return MatchResult(
            found=found,
            confidence=conf,
            center=center if found else None,
            bbox=bbox if found else None,
            template_name=template_name,
        )

    def find(
        self,
        frame: Optional[np.ndarray],
        name: str,
        threshold: float = 0.85,
        roi: Optional[Sequence[float]] = None,
    ) -> MatchResult:
        """Match a cached main template by name against frame."""
        template = self._templates.get(name)
        return self.match_template(
            frame=frame,
            template=template,
            threshold=threshold,
            roi=roi,
            template_name=name,
        )

    def find_dismiss_target(
        self,
        frame: Optional[np.ndarray],
        threshold: float = 0.85,
    ) -> Optional[MatchResult]:
        """Scan optional assets/dismiss/*.png across full frame and return best match >= threshold."""
        if frame is None or not self._dismiss_templates:
            return None

        best: Optional[MatchResult] = None
        for name, template in self._dismiss_templates.items():
            res = self.match_template(
                frame=frame,
                template=template,
                threshold=threshold,
                roi=None,
                template_name=f"dismiss/{name}",
            )
            if res.found and (best is None or res.confidence > best.confidence):
                best = res
        return best

    def save_failure_snapshot(
        self,
        frame: Optional[np.ndarray],
        reason: str,
    ) -> Optional[Path]:
        """Save a timestamped failure PNG and prune older snapshots beyond max_failure_snapshots."""
        if frame is None or frame.size == 0:
            return None

        self.failures_dir.mkdir(parents=True, exist_ok=True)
        safe_reason = re.sub(r"[^a-zA-Z0-9_-]+", "_", reason.strip().lower()).strip("_") or "failure"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target = self.failures_dir / f"{stamp}_{safe_reason}.png"
        if not _write_png_image(target, frame):
            return None

        self._prune_failure_snapshots()
        return target

    def _prune_failure_snapshots(self) -> None:
        snapshots = sorted(
            self.failures_dir.glob("*.png"),
            key=lambda p: (p.stat().st_mtime_ns, p.name),
        )
        excess = len(snapshots) - self.max_failure_snapshots
        for old_path in snapshots[: max(0, excess)]:
            try:
                old_path.unlink()
            except OSError:
                pass
