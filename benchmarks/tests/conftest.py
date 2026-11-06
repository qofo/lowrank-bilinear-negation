"""
Put the two import roots on sys.path.

The repository root makes the code importable as `benchmarks.src.<package>`, the one
name every module and test uses. `benchmarks/src` is there only for the vendored
OpenCLIP, which is imported as the top-level package `open_clip`.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for path in (REPO_ROOT, os.path.join(REPO_ROOT, "benchmarks", "src")):
    if path not in sys.path:
        sys.path.insert(0, path)
