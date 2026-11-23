"""Entry point for the BTD overnight macro: DPI setup, single-instance guard, hotkey, and UI."""

from __future__ import annotations

import queue
import sys
from datetime import datetime
from typing import Any

from src.config import get_failures_dir, get_logs_dir, load_config
from src.controller import RobloxController
from src.state_machine import MacroStateMachine
from src.ui import MacroApp
from src.vision import VisionEngine
from src.win32_utils import (
    GlobalHotkeyListener,
    SingleInstanceGuard,
    allow_sleep,
    enable_dpi_awareness,
)


def main() -> int:
    """Initialize DPI awareness, single-instance mutex, FSM, F8 hotkey, and UI."""
    enable_dpi_awareness()

    guard = SingleInstanceGuard()
    if not guard.acquire():
        print("BTD Overnight Macro is already running.", file=sys.stderr)
        return 1

    hotkey_listener: GlobalHotkeyListener | None = None
    fsm: MacroStateMachine | None = None

    try:
        config = load_config()
        get_logs_dir().mkdir(parents=True, exist_ok=True)
        get_failures_dir().mkdir(parents=True, exist_ok=True)

        event_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        vision = VisionEngine(max_failure_snapshots=config.logging.max_failure_snapshots)
        controller = RobloxController(config)
        fsm = MacroStateMachine(
            config=config,
            vision=vision,
            controller=controller,
            event_queue=event_queue,
        )

        def _on_hotkey() -> None:
            if fsm is not None and fsm.is_running:
                fsm.stop(f"Global hotkey {config.hotkey} pressed")
                try:
                    event_queue.put_nowait(
                        {
                            "type": "log",
                            "level": "WARN",
                            "message": f"Emergency kill switch ({config.hotkey}) triggered!",
                            "ts": datetime.now().strftime("%H:%M:%S"),
                        }
                    )
                except queue.Full:
                    pass

        hotkey_listener = GlobalHotkeyListener(config.hotkey, _on_hotkey)
        registered = hotkey_listener.start()
        if not registered:
            event_queue.put_nowait(
                {
                    "type": "log",
                    "level": "WARN",
                    "message": f"Could not register global hotkey {config.hotkey} (may already be in use).",
                    "ts": datetime.now().strftime("%H:%M:%S"),
                }
            )

        app = MacroApp(
            config=config,
            fsm=fsm,
            controller=controller,
            event_queue=event_queue,
            hotkey_listener=hotkey_listener,
        )
        app.mainloop()
        return 0
    finally:
        if hotkey_listener is not None:
            hotkey_listener.stop()
        if fsm is not None:
            if fsm.is_running:
                fsm.stop("Shutdown")
            fsm.close()
        allow_sleep()
        guard.release()


if __name__ == "__main__":
    raise SystemExit(main())
