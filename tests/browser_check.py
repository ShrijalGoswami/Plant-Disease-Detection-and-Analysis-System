"""
Browser level checks for the running Streamlit app.

These cannot be written with Streamlit's AppTest because it does not support
file uploads, so a real headless Chromium browser (Playwright) drives the
page: it uploads files through the actual uploader widget, reads the page
text and prints one line per check. Screenshots are saved to
assets/screenshots/.

This is not collected by pytest on purpose (the file name does not start
with test_) because it needs a running server and the optional playwright
package. Run it by hand:

    pip install playwright && python -m playwright install chromium
    streamlit run app.py --server.headless true     (in another terminal)
    python tests/browser_check.py
"""

import re
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT = Path(__file__).resolve().parents[1]
SAMPLES = PROJECT / "assets" / "samples"
SCREENSHOTS = PROJECT / "assets" / "screenshots"
URL = "http://localhost:8501"
PREDICTION_PATTERN = r"Crop\s+(\S.*?)\s+Condition\s+(.*?)\s+Confidence\s+([\d.]+%)"

failures = 0


def check(name: str, passed: bool, detail: str = "") -> None:
    global failures
    failures += 0 if passed else 1
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def upload(page, path: Path) -> None:
    page.set_input_files("input[type=file]", str(path))
    # The uploader lists the file name once the upload has finished.
    page.wait_for_selector(f"text={path.name}", timeout=30000)
    page.wait_for_timeout(3000)


def page_text(page) -> str:
    return page.locator("body").inner_text()


def open_tab(page, name: str) -> None:
    page.get_by_role("tab", name=name).click()
    page.wait_for_timeout(1500)


def main() -> int:
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    corrupted = Path(tempfile.gettempdir()) / "corrupted_leaf.jpg"
    corrupted.write_bytes(b"this is a text file pretending to be a jpeg image")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1800})
        page.goto(URL, wait_until="networkidle")
        page.wait_for_selector("text=Plant Disease Detection and Analysis System", timeout=60000)
        page.wait_for_timeout(2000)
        page.screenshot(path=str(SCREENSHOTS / "01_home.png"))
        check("home page loads with the model", "Classes:" in page_text(page))

        upload(page, SAMPLES / "Tomato___Late_blight.jpg")
        page.wait_for_selector("text=Step 2: Disease detection", timeout=30000)
        page.wait_for_timeout(1500)
        match = re.search(PREDICTION_PATTERN, page_text(page), re.S)
        check("diseased JPG gives a prediction", bool(match), str(match.groups()) if match else "no prediction found")
        page.screenshot(path=str(SCREENSHOTS / "02_prediction_diseased.png"))

        upload(page, SAMPLES / "Apple___healthy.jpg")
        text = page_text(page)
        match = re.search(PREDICTION_PATTERN, text, re.S)
        check("healthy leaf shows the healthy banner", bool(match) and "Status: Healthy" in text,
              str(match.groups()) if match else "no prediction found")
        page.screenshot(path=str(SCREENSHOTS / "03_prediction_healthy.png"))

        upload(page, corrupted)
        text = page_text(page)
        check("corrupted file shows an error and the app stays up",
              "could not be decoded" in text and "Plant Disease Detection" in text)
        page.screenshot(path=str(SCREENSHOTS / "04_corrupted_file_error.png"))

        for name in ["Potato___Early_blight.jpg", "Corn_(maize)___Common_rust_.jpg",
                     "Grape___Black_rot.jpg", "Pepper,_bell___healthy.jpg"]:
            upload(page, SAMPLES / name)
        # Switching tabs reruns the script; the history must not gain duplicates.
        open_tab(page, "Session analytics")
        open_tab(page, "Detect")
        open_tab(page, "Session analytics")
        text = page_text(page)
        match = re.search(r"Images analysed\s+(\d+)\s+Healthy\s+(\d+)\s+Diseased\s+(\d+)", text, re.S)
        check("six uploads give 6 rows, 2 healthy, 4 diseased, no duplicates after tab switching",
              bool(match) and match.groups() == ("6", "2", "4"), str(match.groups()) if match else "not found")
        check("CSV and TXT download buttons are present",
              "Download history (CSV)" in text and "Download report (TXT)" in text)
        page.screenshot(path=str(SCREENSHOTS / "05_session_analytics.png"))

        open_tab(page, "Model evaluation")
        page.wait_for_timeout(2000)
        text = page_text(page)
        match = re.search(r"Test accuracy\s+([\d.]+%)", text)
        check("evaluation tab shows the test metrics and confusion matrix",
              bool(match) and "Confusion matrix" in text, match.group(1) if match else "not found")
        page.screenshot(path=str(SCREENSHOTS / "06_model_evaluation.png"))

        open_tab(page, "Session analytics")
        page.get_by_role("button", name="Clear history").click()
        page.wait_for_timeout(2500)
        open_tab(page, "Session analytics")  # a rerun returns to the first tab
        check("clear history empties the session", "No predictions yet" in page_text(page))

        browser.close()

    print(f"\n{8 - failures} of 8 checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
