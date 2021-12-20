import sys

from pathlib import Path
file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from oct_segmenter.preprocessing.image_labeling import generate_image_label_wayne

"""
python3 preprocessing-scripts/generate_image_label.py wayne-images/image1.tiff .
"""
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_image_label.py <path/to/tiff> <path/to/ouptut/dir>")
        exit(1)

    image_path = Path(sys.argv[1])
    generate_image_label_wayne(image_path, sys.argv[2])