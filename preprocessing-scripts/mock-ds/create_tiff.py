"""
Helper script to create TIFF files.

Example:
    python preprocessing-scripts/mock-ds/create_tiff.py
"""

import numpy as np
from PIL import Image

data = np.random.randint(0, 255, (2, 2)).astype(np.uint8)
im = Image.fromarray(data)
im.save("test.tiff")
