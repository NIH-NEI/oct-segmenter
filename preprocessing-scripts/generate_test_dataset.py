import sys

from pathlib import Path
file = Path(__file__). resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from oct_segmenter.preprocessing.test_dataset import generate_test_dataset

"""
python3 preprocessing-scripts/generate_test_dataset.py wayne-images/test wayne-images/wayne_test_dataset.hdf5
"""
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 generate_training_dataset.py <path/to/training/input/dir> <output_file_name>")
        exit(1)

    test_input_dir = sys.argv[1]
    output_file = sys.argv[2]

    generate_test_dataset(test_input_dir, output_file, wayne_format=True)