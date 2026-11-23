"""Unit and smoke tests for the persistent Windows.Media.Ocr worker."""

from __future__ import annotations

import cv2
import numpy as np

from src.ocr import WindowsOcrEngine


def _render_text_card(lines: list[str], w: int = 340, h: int = 120) -> np.ndarray:
    img = np.full((h, w, 3), (38, 30, 24), dtype=np.uint8)
    for idx, text in enumerate(lines):
        cv2.putText(
            img,
            text,
            (18, 36 + idx * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return img


def test_windows_ocr_reads_rendered_upgrade_and_spawner_text() -> None:
    engine = WindowsOcrEngine(timeout_sec=6.0, startup_timeout_sec=15.0)
    try:
        board_img = _render_text_card(["Board V", "+1 Board Size"])
        lines = engine.read_lines(board_img, invert=True, scale=2.0)
        joined = " ".join(lines).lower()
        assert "board" in joined

        buy_img = _render_text_card(["Buy 1,000"], w=260, h=64)
        buy_lines = engine.read_lines(buy_img, invert=True, scale=2.0)
        assert any("buy" in ln.lower() for ln in buy_lines)

        sold_img = _render_text_card(["Sold out"], w=260, h=64)
        sold_lines = engine.read_lines(sold_img, invert=True, scale=2.0)
        assert any("sold" in ln.lower() for ln in sold_lines)
    finally:
        engine.close()


def test_windows_ocr_handles_empty_inputs_and_restart_after_timeout() -> None:
    engine = WindowsOcrEngine(timeout_sec=0.3, startup_timeout_sec=5.0)
    try:
        assert engine.read_lines(None) == []
        assert engine.read_lines(np.zeros((0, 0, 3), dtype=np.uint8)) == []
    finally:
        engine.close()

    # Worker that sleeps longer than timeout_sec to test non-hanging timeout recovery
    slow_script = r"""
[Console]::Out.WriteLine("READY")
[Console]::Out.Flush()
while ($null -ne ($line = [Console]::In.ReadLine())) {
    if ($line -eq "__EXIT__") { break }
    Start-Sleep -Seconds 2
    [Console]::Out.WriteLine("[]")
    [Console]::Out.Flush()
}
"""
    slow_engine = WindowsOcrEngine(timeout_sec=0.25, startup_timeout_sec=5.0, script_override=slow_script)
    try:
        sample = _render_text_card(["Test"])
        assert slow_engine.read_lines(sample) == []
    finally:
        slow_engine.close()
        # Safe to call close multiple times
        slow_engine.close()
