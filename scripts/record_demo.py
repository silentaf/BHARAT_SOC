"""Record the laptop scenes of the demo video (docs/demo_script.md) as silent 1080p clips.

    python scripts/record_demo.py            # -> outputs/demo_video/s1_readme.mp4, s3_inspect.mp4, ...
    python scripts/record_demo.py --only s4_ask_report

Drives the system Google Chrome with Playwright (pip install playwright; no browser download):
starts scripts/demo.py, clicks and types like an operator, and records each scene in its own
clip. Add the voice-over from the script in a video editor. README/result scenes are recorded
from the GitHub page, so they need internet; the dashboard scenes do not.
"""

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
# Playwright's own ffmpeg (video capture) lives in the project: python -m playwright install ffmpeg
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".cache" / "ms-playwright"))
OUT = ROOT / "outputs" / "demo_video"
REPO = "https://github.com/silentaf/BHARAT_SOC"
SIZE = {"width": 1920, "height": 1080}
PORT = 8765  # not 8000: often taken by other local apps
DASH = f"http://127.0.0.1:{PORT}"


def start_demo(scene: str) -> tuple[subprocess.Popen, str | None]:
    """Start scripts/demo.py <scene>; returns the process and (for teach) the folder to paste."""
    p = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "demo.py"), scene, "--port", str(PORT)], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                         env={**os.environ, "PYTHONUNBUFFERED": "1"},
                         start_new_session=True)  # own process group: stop() also ends the dashboard child
    folder, deadline = None, time.time() + 300
    while time.time() < deadline:
        line = p.stdout.readline()
        if not line and p.poll() is not None:
            raise SystemExit(f"demo.py {scene} exited")
        if scene == "teach" and folder is None and line.strip().startswith("/"):
            folder = line.strip()
        if "open http://" in line:
            break
    for _ in range(120):  # our dashboard up (check the page, not just the port)
        try:
            if b'id="advs"' in urllib.request.urlopen(DASH, timeout=2).read():
                break
        except OSError:
            pass
        time.sleep(1)
    else:
        raise SystemExit(f"QC-Assist dashboard did not come up on {DASH}")
    # keep draining output so the server never blocks on a full pipe
    threading.Thread(target=lambda: [None for _ in p.stdout], daemon=True).start()
    return p, folder


def stop(p: subprocess.Popen) -> None:
    """End demo.py and the dashboard.py it started (same process group), so the port is free."""
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(10)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for _ in range(20):
        try:
            urllib.request.urlopen(DASH, timeout=1)
            time.sleep(0.5)
        except OSError:
            return


def record(browser, name: str, fn) -> None:
    """Run fn(page) in a fresh context with video on; save outputs/demo_video/<name>.mp4."""
    tmp = OUT / f"_{name}"
    shutil.rmtree(tmp, ignore_errors=True)
    ctx = browser.new_context(viewport=SIZE, record_video_dir=str(tmp), record_video_size=SIZE)
    page = ctx.new_page()
    fn(page)
    video = page.video.path()
    ctx.close()
    mp4 = OUT / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-r", "30", str(mp4)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  {mp4.relative_to(ROOT)}")


def open_page(page, url: str) -> None:
    """GitHub pages keep loading trackers; the content is there long before the load event."""
    page.goto(url, wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_timeout(2500)


def scroll_to(page, text: str, pause: float) -> None:
    page.get_by_role("heading", name=re.compile(text)).first.scroll_into_view_if_needed()
    page.evaluate("window.scrollBy({top: -80})")
    page.wait_for_timeout(pause * 1000)


def slow_scroll(page, pixels: int, seconds: float) -> None:
    steps = int(seconds * 10)
    for _ in range(steps):
        page.evaluate(f"window.scrollBy(0, {pixels / steps})")
        page.wait_for_timeout(100)


def wait_text_change(page, sel: str, old: str, timeout_s: float) -> None:
    end = time.time() + timeout_s
    while time.time() < end:
        t = page.inner_text(sel).strip()
        if t and t != old and not t.endswith("…") and "thinking" not in t.lower():
            return
        page.wait_for_timeout(500)


# ---------- scenes ----------

def s1_readme(page):
    open_page(page, REPO)
    page.wait_for_timeout(2000)
    scroll_to(page, "QC-Assist", 6)
    slow_scroll(page, 700, 8)


def s2_architecture(page):
    open_page(page, REPO)
    scroll_to(page, "Architecture", 2)
    slow_scroll(page, 450, 18)


def s3_inspect(page):
    page.goto(DASH)
    page.wait_for_timeout(75_000)


def s4_ask_report(page):
    page.goto(DASH)
    page.wait_for_timeout(3000)
    page.click("#q")
    page.type("#q", "Where are most defects this shift?", delay=60)
    old = page.inner_text("#answer")
    page.click("#ask")
    wait_text_change(page, "#answer", old, 60)
    ok = "(llama-executorch" in page.inner_text("#answer")
    page.wait_for_timeout(8000)
    old = page.inner_text("#answer")
    page.click("#report")
    wait_text_change(page, "#answer", old, 90)
    TAKE["llama_ok"] = ok and "(llama-executorch" in page.inner_text("#answer")
    page.wait_for_timeout(12_000)


TAKE = {}


def s5_hindi(page):
    page.goto(DASH)
    page.wait_for_timeout(3000)
    page.click("#lang")
    old = page.inner_text("#advs")
    wait_text_change(page, "#advs", old, 90)
    page.wait_for_timeout(8000)
    old = page.inner_text("#answer")
    page.click("#report")
    wait_text_change(page, "#answer", old, 90)
    page.wait_for_timeout(12_000)
    page.click("#lang")  # back to English for later scenes


def s6_teach(folder):
    def run(page):
        page.goto(DASH)
        page.wait_for_timeout(5000)
        page.click("#folder")
        page.type("#folder", folder, delay=15)
        page.wait_for_timeout(1500)
        page.click("#teachf")
        page.wait_for_timeout(30_000)
    return run


def s7_results(page):
    open_page(page, f"{REPO}/blob/main/results/x86_64/summary.md")
    page.wait_for_timeout(9000)
    open_page(page, f"{REPO}/blob/main/docs/hailo.md")
    page.wait_for_timeout(2000)
    slow_scroll(page, 900, 12)
    open_page(page, f"{REPO}#results-laptop-and-an-arm-phone-below")
    page.wait_for_timeout(2000)
    slow_scroll(page, 600, 10)


def s8_status(page):
    open_page(page, REPO)
    scroll_to(page, "Status", 15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="scene names, e.g. s3_inspect s6_teach")
    args = ap.parse_args()
    want = lambda n: not args.only or n in args.only  # noqa: E731
    OUT.mkdir(parents=True, exist_ok=True)
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=chrome, headless=True)
        for name, fn in [("s1_readme", s1_readme), ("s2_architecture", s2_architecture)]:
            if want(name):
                record(browser, name, fn)
        if any(want(n) for n in ("s3_inspect", "s4_ask_report", "s5_hindi")):
            demo, _ = start_demo("inspect")
            try:
                warm = browser.new_context()  # first Llama call loads the model: not on camera
                warm.new_page().goto(DASH)
                warm.pages[0].wait_for_timeout(45_000)
                warm.close()
                for name, fn in [("s3_inspect", s3_inspect), ("s4_ask_report", s4_ask_report), ("s5_hindi", s5_hindi)]:
                    if not want(name):
                        continue
                    for take in range(3 if name == "s4_ask_report" else 1):
                        record(browser, name, fn)
                        if name != "s4_ask_report" or TAKE.get("llama_ok"):
                            break
                        print(f"  take {take + 1}: a Llama answer was replaced by the checked figures; retaking")
            finally:
                stop(demo)
        if want("s6_teach"):
            demo, folder = start_demo("teach")
            try:
                record(browser, "s6_teach", s6_teach(folder))
            finally:
                stop(demo)
        for name, fn in [("s7_results", s7_results), ("s8_status", s8_status)]:
            if want(name):
                record(browser, name, fn)
        browser.close()


if __name__ == "__main__":
    main()
