"""Capture dashboard screenshots with a real browser.

Usage (dashboard must already be running)::

    .venv\\Scripts\\python.exe reports\\capture_screenshots.py --url http://127.0.0.1:8512 \\
        --out reports/figures

The script drives the installed Chrome/Edge via Playwright (``channel="chrome"``),
clicks through every tab and writes a PNG per view. It is a development helper,
not part of the package: it needs a browser and a running server, so it is kept
out of the test suite and out of the runtime dependencies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

TABS = [
    ("overview", "Overview"),
    ("metrics_anomalies", "Metrics & anomalies"),
    ("drift", "Drift"),
    ("alerts", "Alerts"),
    ("evaluation", "Evaluation"),
    ("provenance", "Provenance"),
]


def capture(page: Page, out: Path, width: int, height: int) -> list[Path]:
    """Click through every tab and screenshot it.

    Streamlit scrolls inside an inner container, so a tall viewport is used
    instead of ``full_page=True`` (the document body itself has no height).
    """
    written: list[Path] = []
    page.set_viewport_size({"width": width, "height": height})
    page.wait_for_selector("text=StreamGuard", timeout=60_000)
    page.wait_for_timeout(8_000)
    for slug, label in TABS:
        page.get_by_role("tab", name=label, exact=True).click()
        page.wait_for_timeout(3_500)
        target = out / f"dashboard_{slug}.png"
        page.screenshot(path=str(target))
        written.append(target)
        print(f"captured {label} -> {target}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8512")
    parser.add_argument("--out", default="reports/figures")
    parser.add_argument("--channel", default="chrome", help="chrome, msedge, or empty for bundled chromium")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=2200)
    args = parser.parse_args(argv)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel or None, headless=True)
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.goto(args.url, wait_until="load", timeout=60_000)
        written = capture(page, output_dir, args.width, args.height)
        browser.close()
    print(f"\n{len(written)} screenshots written to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover - development helper
    sys.exit(main())
