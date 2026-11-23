# BTD Overnight Macro — live test results and fixes (2026-11-23)

Handoff for the next agent. Covers the live test of the Auto-Spawners / Auto-Upgrade build, the bugs it found, and what was changed. Project root: `btd-macro/`. Not a git repo; backups of pre-change files are noted at the end.

## Current state

- `pytest`: **69/69 pass** (was 60 at the start of the session).
- Live, on the latest code: a full Spawner pass **bought out the whole shop** (both pages showed every card Sold out afterwards). Order was correct: `PURCHASING_SPAWNERS` → `UPGRADING` → Battle/Continue → `IN_RUN`.
- Macro UI is reloaded on the final code and idle. Roblox is in the lobby.

## Original test plan results

| Check | Result |
|---|---|
| Pre-checks (config, templates) | PASS after capturing the two missing templates |
| Template confidence | PASS: Spawners 0.906 / 0.911 (normal / enlarged), SpawnersClose 1.000 with the shop open |
| Dry-run | PASS: correct pill count/positions; upgrade scan logged `start_h`, `cov`, `clipped` |
| Board guard | PASS: no Board upgrade bought live. Hover test: Board V tooltip → `blocked (keyword:board)` |
| Priority order | PASS on first cycle; next-cycle bug found and fixed (see 1 below) |
| Spawner purchase | Several bugs found; replaced with a click-all strategy (see 4 below) |
| Upgrade camera drift / hover-before-click | Not visually verified (passes too fast to watch) |

## Bugs found and fixed

1. **Runs ending via "Wave HUD missing" were never counted.** `IN_RUN` only counted a run when Battle reappeared *while in IN_RUN*. When the HUD vanished first (end-of-run screen), it fell back to `READY_TO_START` uncounted, so no Spawner/Upgrade pass was queued and Battle was clicked straight away.
   *Fix:* `_run_end_unconfirmed` flag set on the HUD-missing fallback; if Battle then appears in `READY_TO_START`, `_record_run_completed()` counts it and queues both passes. The flag clears on any transition back to `IN_RUN` (mid-run HUD dropouts such as milestone banners don't count). Verified live: `Run #1 completed (Battle button appeared after wave HUD was lost)` followed by the spawner pass.

2. **Spawner pass ended early and left the shop open.** The "+1 <Spawner>!" coin-burst animation covers the close X; one missed match was treated as "shop closed", the pass ended, and `READY_TO_START` stalled behind the open shop until the watchdog.
   *Fix:* `_confirm_modal_absent()` requires the close X to be missing across several frames, in both the pass rescan and `_close_modal_via_template`. A safety net in `READY_TO_START` closes a Spawner Shop / Upgrade Tree left open.

3. **Opening a modal clicked only once.** The first click after Roblox regains focus is often ignored, which skipped the whole pass.
   *Fix:* `_open_modal_via_template` hovers first and tries up to 3 times.

4. **Spawner buying was unreliable (main user complaint).** Frame recording showed a purchase takes ~0.5 s (longer at high ping) to show on the card. The old code rescanned after 0.25 s and judged success by "card area OR cash header changed"; cash rises passively, so it logged purchases that hadn't happened and clicked again mid-purchase. OCR checks also rejected valid pills ("Buy 1,800" read as `Buya,800` / `BG7,500` / empty under the coin burst or cursor hover).
   *Fix (user's design):* **click every green Buy pill, no reading or confirming.** Per page: click all green pills row-major (shop is sorted cheapest → most expensive), wait, rescan, repeat until none are green or `clicks_per_pill` rounds; if pills are still green after the last round, assume out of cash and stop (later pages cost more); then scroll. Stops after 3 scroll steps in a row with no green pills (animations defeat the grid-unchanged check). Sold out / Robux pills are never green, so they're never clicked. Each click hovers `click_hover_sec` first.

5. **A covered Roblox window triggered a relaunch.** Capture is screen-based; when another window (e.g. the chat app) covered Roblox, `READY_TO_START` saw nothing and the watchdog relaunched Roblox after 120 s.
   *Fix:* if nothing is recognised and Roblox isn't the foreground window, the macro refocuses it (throttled to once per 5 s). `RobloxController.is_focused()` added.

6. **Templates.** Captured `assets/btn_spawners.png` and `assets/btn_spawners_close.png`. After the shop has been opened, the lobby Spawners button stays ~5% larger, and a tight crop scored only 0.70. The template is the icon crop scaled 1.03×, which scores ≥0.90 in both states.

## Files changed

- `src/state_machine.py`: run-completion flag and `_record_run_completed`; `_confirm_modal_absent`; open retry with hover; `READY_TO_START` leftover-modal close and refocus; Spawner pass rewritten to click-all.
- `src/controller.py`: `is_focused()`.
- `src/config.py` + `config.json`, `auto_spawner`:
  - New: `clicks_per_pill=5`, `click_hover_sec=0.12`, `click_interval_sec=0.35`, `round_settle_sec=1.0`.
  - Changed: `max_purchases_per_pass` 50→80, `pass_timeout_sec` 45→90.
  - Removed: `post_click_settle_sec`, `verify_buy_text_ocr` (spawner only).
  - `timeouts.purchasing_spawners_sec` 60→120.
- `src/ui.py`: stat label "Spawners Bought" → "Spawner Buy Clicks" (the counter now counts clicks, not confirmed buys).
- `src/spawner_vision.py`: `_BUY_START_RE` no longer needs a word boundary; added `parse_spawner_stock` and a `check_cash` option on `did_spawner_purchase_register`. **These and `is_buy_pill_text_valid` / `extract_card_stock_patch` are no longer used by the state machine** (only by tests) and could be removed.
- Tests: `tests/test_state_machine.py` (new `_FakeShop` simulator and regression tests for each fix), `tests/test_config.py`, `tests/test_spawner_vision.py`.

## Known issues / not done

- **Laggy servers and Battle start:** at ~742 ms ping, Battle clicks didn't start a battle within `click_confirm_timeout_sec` (3 s), and after 5 tries the watchdog relaunched Roblox (`battle_click_unconfirmed`). The relaunch landed on a 30 ms server, so it recovered, but it's an unexpected restart. Consider a longer confirm window or more retries before a relaunch.
- **`hud_in_run.png` matches only the red STOP button.** With the game's "classic forfeit" setting on, the HUD shows FORFEIT instead and the macro can't see an active run. Keep that setting off, or add a template that works in both modes.
- **Upgrade pass:** one log showed `[pan-bottom] Tree coverage dropped to 0.03 without Start anchor; aborting pan.` It aborted safely; drift wasn't visually verified.
- **Spawner Buy Clicks** overstates purchases when out of cash (up to `clicks_per_pill` clicks per unaffordable pill before the pass stops).

## Testing tips

- To end a run early: Stop the macro, press the in-game STOP, then Confirm (battle is saved; discard via the Continue prompt).
- The macro's Start button needs a slow press/release when driven by automation; quick clicks don't register.
- Roblox may run from Bloxstrap's folder (`%LOCALAPPDATA%\Bloxstrap\Versions\...`) after a relaunch.

## Backups (scratchpad, may be cleared)

Pre-change copies of `state_machine.py`, `config.py`, `config.json`, `ui.py` and the test files were kept in the session scratchpad (`*.orig`, `*.pre_buyfix`, `*.pre_clickall`).
