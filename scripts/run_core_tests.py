"""Run the self-contained deterministic core tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TEST_MODULES = (
    "tests.test_data",
    "tests.test_ir_v2",
    "tests.test_solver",
    "tests.test_validator",
    "tests.test_methods_v2",
    "tests.test_cache_metrics",
    "tests.test_end_to_end",
    "tests.test_gold",
    "tests.test_semantic_v2_2",
    "tests.test_patch_parsing_ir",
)


def main() -> int:
    """Load the fixed core suite and return a process status."""
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root))

    suite = unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
