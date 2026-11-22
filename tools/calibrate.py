"""Interactive calibration tool to snap Roblox, crop templates, and verify live confidences."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import get_assets_dir, get_dismiss_dir, load_config
from src.controller import RobloxController
from src.vision import (
    OPTIONAL_TEMPLATES,
    REQUIRED_TEMPLATES,
    VisionEngine,
    _read_bgr_image,
    _write_png_image,
)
from src.win32_utils import enable_dpi_awareness


def snap_roblox(controller: RobloxController) -> Optional[tuple[int, int, int, int]]:
    """Snap an open RobloxPlayerBeta.exe window to configured layout and return client rect."""
    hwnd = controller.get_hwnd()
    if hwnd is None:
        print("[ERROR] No visible RobloxPlayerBeta.exe window found.")
        return None
    if not controller.ensure_window_layout(hwnd):
        print("[ERROR] Failed to reposition/resize Roblox window.")
        return None
    controller.focus(hwnd)
    time.sleep(0.2)
    rect = controller.get_client_rect(hwnd)
    if rect is None:
        print("[ERROR] Could not query Roblox client rectangle.")
        return None
    w_cfg = controller.config.window
    print(
        f"[OK] Snapped Roblox window to {w_cfg.width}x{w_cfg.height} at ({w_cfg.x}, {w_cfg.y}). "
        f"Client rect: left={rect[0]}, top={rect[1]}, width={rect[2]}, height={rect[3]}"
    )
    return rect


def _capture_source_frame(
    controller: RobloxController,
    vision: VisionEngine,
    from_image: Optional[Path] = None,
) -> Optional[np.ndarray]:
    if from_image is not None:
        img = _read_bgr_image(from_image)
        if img is None:
            print(f"[ERROR] Failed to read image: {from_image}")
        return img

    rect = snap_roblox(controller)
    if rect is None:
        return None
    frame = vision.capture_client(rect)
    if frame is None:
        print("[ERROR] Failed to capture Roblox client area.")
    return frame


def crop_template_interactive(
    name: str,
    output_path: Path,
    controller: RobloxController,
    vision: VisionEngine,
    from_image: Optional[Path] = None,
) -> bool:
    """Open cv2.selectROI on a clean client capture (or image) and save the cropped template."""
    frame = _capture_source_frame(controller, vision, from_image=from_image)
    if frame is None:
        return False

    print(
        f"\nSelect tight opaque region for '{name}' (avoid translucent edges).\n"
        "Drag box with mouse, then press ENTER or SPACE to confirm, or C to cancel."
    )
    win_title = f"Crop Template: {name} (ENTER/SPACE=Save, C=Cancel)"
    roi = cv2.selectROI(win_title, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win_title)

    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        print(f"[SKIP] Cancelled cropping '{name}'.")
        return False

    crop = np.ascontiguousarray(frame[y : y + h, x : x + w])
    if not _write_png_image(output_path, crop):
        print(f"[ERROR] Could not write template to {output_path}")
        return False

    print(f"[SAVED] {output_path} ({w}x{h} px)")
    vision.reload_templates()
    return True


def run_confidence_monitor(
    controller: RobloxController,
    vision: VisionEngine,
    interval_sec: float = 0.5,
) -> None:
    """Print live BGR template confidence scores without clicking anything."""
    vision.reload_templates()
    missing = vision.missing_required_templates()
    if missing:
        print(f"[WARN] Missing templates: {', '.join(missing)}")

    cfg = controller.config
    print("\nLive Dry-Run Confidence Monitor started (Ctrl+C to stop)...")
    try:
        while True:
            rect = controller.get_client_rect()
            if rect is None:
                print("[WAIT] Roblox window not visible or minimized...")
                time.sleep(interval_sec)
                continue

            frame = vision.capture_client(rect)
            if frame is None:
                print("[WARN] Failed to capture client frame.")
                time.sleep(interval_sec)
                continue

            battle = vision.find(frame, "btn_battle", cfg.thresholds.btn_battle, cfg.rois.btn_battle)
            disc = vision.find(
                frame,
                "modal_disconnected",
                cfg.thresholds.modal_disconnected,
                cfg.rois.modal_disconnected,
            )
            recon = vision.find(
                frame,
                "btn_reconnect",
                cfg.thresholds.btn_reconnect,
                cfg.rois.btn_reconnect,
            )
            resume_name = (
                "btn_start_over" if cfg.recovery.resume_mode == "start_over" else "btn_yes_continue"
            )
            resume_th = (
                cfg.thresholds.btn_start_over
                if resume_name == "btn_start_over"
                else cfg.thresholds.btn_yes_continue
            )
            resume_roi = (
                cfg.rois.btn_start_over
                if resume_name == "btn_start_over"
                else cfg.rois.btn_yes_continue
            )
            resume = vision.find(frame, resume_name, resume_th, resume_roi)
            cont = vision.find(
                frame,
                "btn_continue",
                cfg.thresholds.btn_continue,
                cfg.rois.btn_continue,
            )
            in_run = vision.find(
                frame,
                "hud_in_run",
                cfg.thresholds.hud_in_run,
                cfg.rois.hud_in_run,
            )
            dismiss = vision.find_dismiss_target(frame, cfg.thresholds.dismiss)

            parts = [
                f"Battle={battle.confidence:.3f}{'*' if battle.found else ''}",
                f"InRun={in_run.confidence:.3f}{'*' if in_run.found else ''}",
                f"Resume={resume.confidence:.3f}{'*' if resume.found else ''}",
                f"Disc={disc.confidence:.3f}{'*' if disc.found else ''}",
                f"Recon={recon.confidence:.3f}{'*' if recon.found else ''}",
            ]
            if vision.has_template("btn_continue"):
                parts.append(f"Continue={cont.confidence:.3f}{'*' if cont.found else ''}")
            if dismiss is not None:
                parts.append(f"{dismiss.template_name}={dismiss.confidence:.3f}*")
            print(" | ".join(parts))
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\n[STOPPED] Confidence monitor stopped.")


def _interactive_menu(controller: RobloxController, vision: VisionEngine) -> int:
    assets_dir = get_assets_dir()
    dismiss_dir = get_dismiss_dir()
    single_targets = {
        "3": "btn_battle",
        "4": "modal_disconnected",
        "5": "btn_reconnect",
        "6": "hud_in_run",
        "7": "btn_yes_continue",
        "8": "btn_continue",
        "9": "btn_start_over",
    }

    while True:
        print("\n=== BTD Macro Calibration Tool ===")
        print("1. Snap Roblox window to 1280x720 at (0, 0)")
        print("2. Capture & crop all 3 required templates (guided)")
        print("3. Capture & crop btn_battle.png")
        print("4. Capture & crop modal_disconnected.png")
        print("5. Capture & crop btn_reconnect.png")
        print("6. Capture & crop hud_in_run.png        (Recommended: red STOP button / wave HUD)")
        print("7. Capture & crop btn_yes_continue.png  (Recommended mid-run resume button)")
        print("8. Capture & crop btn_continue.png      (Optional bottom Continue button)")
        print("9. Capture & crop btn_start_over.png    (Optional Start Over button)")
        print("10. Capture & crop optional dismiss/*.png popup")
        print("11. Live Dry-Run Confidence Monitor")
        print("12. Exit")

        choice = input("Select option [1-12]: ").strip()
        if choice == "1":
            snap_roblox(controller)
        elif choice == "2":
            for name in REQUIRED_TEMPLATES:
                ans = input(
                    f"\nPrepare Roblox screen with '{name}' visible.\n"
                    "Press ENTER to capture live, enter a screenshot path, or 's' to skip: "
                ).strip()
                if ans.lower() == "s":
                    continue
                src = Path(ans.strip('"')) if ans else None
                crop_template_interactive(name, assets_dir / f"{name}.png", controller, vision, src)
        elif choice in single_targets:
            name = single_targets[choice]
            ans = input("Press ENTER to capture live Roblox, or enter a screenshot path: ").strip()
            src = Path(ans.strip('"')) if ans else None
            crop_template_interactive(name, assets_dir / f"{name}.png", controller, vision, src)
        elif choice == "10":
            popup_name = input("Enter filename for dismiss template (e.g., popup_close): ").strip()
            if popup_name:
                ans = input("Press ENTER to capture live Roblox, or enter a screenshot path: ").strip()
                src = Path(ans.strip('"')) if ans else None
                crop_template_interactive(
                    f"dismiss/{popup_name}",
                    dismiss_dir / f"{popup_name}.png",
                    controller,
                    vision,
                    src,
                )
        elif choice == "11":
            run_confidence_monitor(controller, vision)
        elif choice == "12":
            return 0
        else:
            print("Invalid choice.")


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for Roblox window snapping, template cropping, and confidence testing."""
    enable_dpi_awareness()

    all_templates = list(REQUIRED_TEMPLATES + OPTIONAL_TEMPLATES)
    parser = argparse.ArgumentParser(description="Calibrate BTD macro templates and window layout.")
    parser.add_argument("--snap", action="store_true", help="Snap Roblox window to configured size/pos.")
    parser.add_argument("--monitor", action="store_true", help="Run live confidence monitor.")
    parser.add_argument(
        "--capture",
        choices=all_templates,
        help="Capture and crop a required or fallback template.",
    )
    parser.add_argument("--dismiss", type=str, help="Capture and crop an optional dismiss template.")
    parser.add_argument("--from-image", type=Path, help="Optional PNG file to crop from instead of live screen.")
    args = parser.parse_args(argv)

    config = load_config()
    controller = RobloxController(config)
    vision = VisionEngine(max_failure_snapshots=config.logging.max_failure_snapshots)

    if args.snap:
        return 0 if snap_roblox(controller) is not None else 1
    if args.capture:
        ok = crop_template_interactive(
            args.capture,
            get_assets_dir() / f"{args.capture}.png",
            controller,
            vision,
            from_image=args.from_image,
        )
        return 0 if ok else 1
    if args.dismiss:
        ok = crop_template_interactive(
            f"dismiss/{args.dismiss}",
            get_dismiss_dir() / f"{args.dismiss}.png",
            controller,
            vision,
            from_image=args.from_image,
        )
        return 0 if ok else 1
    if args.monitor:
        run_confidence_monitor(controller, vision)
        return 0

    return _interactive_menu(controller, vision)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
