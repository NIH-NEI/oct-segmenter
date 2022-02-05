import sys

from pathlib import Path
file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from oct_segmenter.preprocessing.generic_dataset import generate_generic_dataset

"""
python3 preprocessing-scripts/generate_dataset.py wayne-images/ wayne-images/wayne_images.hdf5
"""
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python generate_dataset.py </path/to/input/dir> </path/to/output/hdf5/file>")
        exit(1)

    input_dir = Path(sys.argv[1])
    hdf5_file_name = Path(sys.argv[2])
    generate_generic_dataset(
        input_dir,
        hdf5_file_name,
        input_format="wayne",
    )
