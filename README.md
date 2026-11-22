# btd-macro

A lightweight Windows overnight automation macro for Roblox *Build a Tower Defense* (Place ID: `113140456364856`).

## Features

- **Fixed-Size DPI-Aware Windowing:** Automatically launches or snaps Roblox to `1280x720` at `(0, 0)` with DPI awareness enabled so template matching stays 1:1 scale-accurate.
- **Fast Local Vision:** Uses `mss` client-area capture and OpenCV (`cv2`) BGR template matching (`TM_CCOEFF_NORMED`) within configured ROIs.
- **Overnight Resilience:** Detects disconnects and non-reconnectable modals, handles Roblox updates with an extended grace period, retries clicks with confirmation checks, and enforces a circuit breaker on repeated failures.
- **Anti-AFK & Sleep Prevention:** Sends periodic anti-AFK input and prevents Windows from sleeping while the macro is running.
- **Failure Diagnostics:** Logs session activity and saves failure snapshots in `logs/failures/`.
- **Desktop UI & Kill Switch:** CustomTkinter control panel with an instant `F8` global emergency stop hotkey.

## Requirements

- Windows 10 / 11
- Python 3.10+
- Roblox Player (`RobloxPlayerBeta.exe`)

## Installation

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Calibration & Usage

1. In *Build a Tower Defense*, enable the **Auto-Close Results** setting.
2. Run `calibrate.bat` to verify or re-crop UI templates in `assets/` (see `assets/README.txt` for details).
3. Run `run.bat` to launch the macro control panel.
4. Press `F8` at any time to trigger the emergency kill switch.

## Configuration

Settings are stored in `config.json`, including:
- `place_id` and `window` layout (`1280x720` at `(0, 0)`)
- `thresholds` and normalized `rois` for template matching
- `polling` intervals and state `timeouts`
- `anti_afk` interval and key
- `recovery` limits and `resume_mode`
- `ui` panel dimensions and global `hotkey`

## Running Tests

```bat
.venv\Scripts\python.exe -m pytest
```

## License

MIT License — see [LICENSE](LICENSE) for details.
