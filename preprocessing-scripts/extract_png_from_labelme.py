import sys

import json
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from oct_segmenter.common import utils


if __name__ == "__main__":
    """
    python3 preprocessing-scripts/extract_png_from_labelme.py </path/to/input/labelme/file> <path/to/ouptut/png>
    """
    if len(sys.argv) != 3:
        print(
            "Usage: python extract_png_from_labelme.py </path/to/input/labelme/file> <path/to/ouptut/png>"
        )
        exit(1)

    img_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(img_path) as f:
        data = json.load(f)
        img = utils.img_b64_to_pil(data["imageData"])
        img.save(output_path)
