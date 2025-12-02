#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Import 테스트"""

print("Testing imports...")

try:
    import sys
    print("✓ sys")
except Exception as e:
    print(f"✗ sys: {e}")

try:
    import json
    print("✓ json")
except Exception as e:
    print(f"✗ json: {e}")

try:
    from pathlib import Path
    print("✓ pathlib")
except Exception as e:
    print(f"✗ pathlib: {e}")

try:
    from collections import Counter
    print("✓ collections")
except Exception as e:
    print(f"✗ collections: {e}")

try:
    import numpy as np
    print("✓ numpy")
except Exception as e:
    print(f"✗ numpy: {e}")

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    print("✓ matplotlib")
except Exception as e:
    print(f"✗ matplotlib: {e}")

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    print("✓ matplotlib Qt5 backend")
except Exception as e:
    print(f"✗ matplotlib Qt5 backend: {e}")

try:
    from PyQt5.QtWidgets import QApplication
    print("✓ PyQt5")
except Exception as e:
    print(f"✗ PyQt5: {e}")

try:
    from PIL import Image
    print("✓ PIL")
except Exception as e:
    print(f"✗ PIL: {e}")

print("\nAll imports tested!")
