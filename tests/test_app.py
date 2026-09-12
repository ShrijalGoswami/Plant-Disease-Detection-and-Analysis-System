"""
Basic application behaviour, using Streamlit's built in AppTest runner.

AppTest executes app.py the same way `streamlit run` does, without a
browser, and records any exception raised by the script. It needs the
trained model files, so the test is skipped when they are absent.
"""

import pytest
from streamlit.testing.v1 import AppTest

from config import CLASS_NAMES_PATH, PROJECT_ROOT, WEIGHTS_PATH

needs_model = pytest.mark.skipif(
    not (WEIGHTS_PATH.exists() and CLASS_NAMES_PATH.exists()),
    reason="trained weights not present, run training/train.py first",
)


@needs_model
def test_app_starts_without_errors():
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=120)
    app.run()
    assert not app.exception, app.exception
    assert app.title[0].value == "Plant Disease Detection and Analysis System"
    # Three tabs, and the detect tab asks for an image before anything else.
    assert len(app.tabs) == 3
    assert any("Waiting for an image" in info.value for info in app.info)
    # Session history starts empty.
    assert app.session_state["history"] == []
