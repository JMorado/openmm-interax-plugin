# Part of the OpenMM InteraxFF plugin. Copyright (c) 2026 the Authors. MIT license, see LICENSE.
"""
The comparisons with InteraxFF need an InteraxFF checkout (INTERAX_REPO) and a model (INTERAX_MODEL): without them
they are not collected, and the report header says so (plugin_path.py).
"""
import os

NEEDS_INTERAXFF = ["test_alchemical.py", "test_multipole_features.py"]
_interaxff = bool(os.environ.get("INTERAX_REPO") and os.environ.get("INTERAX_MODEL"))
collect_ignore = [] if _interaxff else NEEDS_INTERAXFF


def pytest_report_header(config):
    if not _interaxff:
        return f"not collected without INTERAX_REPO and INTERAX_MODEL: {', '.join(NEEDS_INTERAXFF)}"
