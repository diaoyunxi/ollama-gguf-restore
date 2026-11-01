"""Root conftest: make the in-tree package importable under pytest.

This lets tests run without a prior ``pip install``.
"""

import os
import sys

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
