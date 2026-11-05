"""Open each Streamlit app in a real headless browser so it counts as traffic.

Usage: python wake_app.py            (reads STREAMLIT_APP_URL, comma-separated)
       python wake_app.py URL [URL]  (explicit URLs)
Exit code is non-zero if any app FAILED. Screenshots go to ./screenshots/.
"""
import os
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

SLEEP_TEXT = "This app has gone to sleep due to inactivity"
WAKE_BUTTON = "Yes, get this app back up!"
APP_SELECTOR = '[data-testid="stAppViewContainer"]'
WAKE_TIMEOUT_S = 180
DWELL_S = int(os.environ.get("DWELL_SECONDS", "18"))
SHOT_DIR = Path(os.environ.get("SCREENSHOT_DIR", "screenshots"))


def is_sleeping(page) -> bool:
    return page.get_by_text(SLEEP_TEXT).count() > 0


def app_loaded(page) -> bool:
    # Streamlit Cloud renders the app inside an iframe, so check every frame.
    return any(f.locator(APP_SELECTOR).count() > 0 for f in page.frames)


def check(page, url: str) -> str:
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    try:
        page.wait_for_load_state("networkidle", timeout=20_000)
    except PWTimeout:
        pass  # Streamlit keeps a websocket open; not fatal

    # Either the sleep screen or the app container should appear.
    deadline = time.time() + 45
    while time.time() < deadline:
        if is_sleeping(page) or app_loaded(page):
            break
        page.wait_for_timeout(1000)
    else:
        raise RuntimeError("neither app nor sleep screen appeared within 45s")

    if is_sleeping(page):
        page.get_by_role("button", name=WAKE_BUTTON).click(timeout=15_000)
        end = time.time() + WAKE_TIMEOUT_S
        while time.time() < end:
            if app_loaded(page) and not is_sleeping(page):
                page.wait_for_timeout(5_000)
                return "WOKEN"
            page.wait_for_timeout(3_000)
        raise RuntimeError(f"app did not load within {WAKE_TIMEOUT_S}s after clicking wake button")

    page.wait_for_timeout(DWELL_S * 1000)
    if is_sleeping(page):
        raise RuntimeError("sleep screen appeared during visit")
    return "AWAKE"


def main(urls):
    failed = 0
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for i, url in enumerate(urls):
            ctx = browser.new_context(viewport={"width": 1280, "height": 800})
            page = ctx.new_page()
            try:
                status = check(page, url)
                print(f"{status}: {url}", flush=True)
            except Exception as e:  # each URL handled independently
                failed += 1
                reason = str(e).splitlines()[0] if str(e) else type(e).__name__
                print(f"FAILED: {url} ({reason})", flush=True)
                shot = SHOT_DIR / f"failed-{i}-{re.sub(r'[^A-Za-z0-9]+', '_', url)[:60]}.png"
                try:
                    page.screenshot(path=str(shot), full_page=True)
                    print(f"  screenshot saved: {shot}", flush=True)
                except Exception as se:
                    print(f"  screenshot failed: {se}", flush=True)
            finally:
                ctx.close()
        browser.close()
    return 1 if failed else 0


if __name__ == "__main__":
    args = sys.argv[1:] or os.environ.get("STREAMLIT_APP_URL", "").split(",")
    urls = [u.strip() for u in args if u.strip()]
    if not urls:
        print("No URLs given (set STREAMLIT_APP_URL).", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(urls))
