"""
The comparisons with the InteraxFF export need InteraxFF (INTERAX_REPO, or installed) and a model (INTERAX_MODEL, or
INTERAX_REPO/examples): without them they are not collected, and the report header says so (plugin_path.py).
"""
import importlib.util
import os
import sys

NEEDS_INTERAXFF = ["test_alchemical.py", "test_multipole_features.py"]
if os.environ.get("INTERAX_REPO"):
    sys.path.insert(0, os.environ["INTERAX_REPO"])
_interaxff = importlib.util.find_spec("interax") is not None and bool(os.environ.get("INTERAX_MODEL") or os.environ.get("INTERAX_REPO"))
collect_ignore = [] if _interaxff else NEEDS_INTERAXFF


def pytest_report_header(config):
    if not _interaxff:
        return f"not collected without InteraxFF and a model (INTERAX_REPO / INTERAX_MODEL): {', '.join(NEEDS_INTERAXFF)}"
