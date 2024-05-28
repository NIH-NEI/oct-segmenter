import sys

import h5py

from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from oct_segmenter.preprocessing.training_dataset import generate_training_dataset

"""
python3 preprocessing-scripts/generate_training_dataset.py wayne-images/training wayne-images/validation wayne-images/wayne_training_dataset.hdf5
"""
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python3 generate_training_dataset.py <path/to/training/input/dir> <path/to/validation/input/dir> <output_file_name>"
        )
        exit(1)

    train_input_dir = Path(sys.argv[1])
    validation_input_dir = Path(sys.argv[2])
    output_file = Path(sys.argv[3])

    training_dataset: h5py.File = generate_training_dataset(
        train_input_dir, validation_input_dir, output_file, input_format="wayne"
    )

    training_dataset.close()
