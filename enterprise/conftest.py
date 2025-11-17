"""
Pytest configuration for enterprise tests.

Ensures enterprise tests import from enterprise/sanitizer instead of root sanitizer.
"""
import sys
from pathlib import Path

# Get the enterprise directory
ENTERPRISE_DIR = Path(__file__).parent

# Insert enterprise directory at the beginning of sys.path
# This makes 'import sanitizer' resolve to enterprise/sanitizer/
if str(ENTERPRISE_DIR) not in sys.path:
    sys.path.insert(0, str(ENTERPRISE_DIR))
