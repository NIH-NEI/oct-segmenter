import os
import shutil
import sys

import numpy as np
from pathlib import Path


TEST_PARTITION = 0.3
TRAINING_PARTITION = 0.8 * (1 - TEST_PARTITION)
VALIDATION_PARTITION = 0.2 * (1 - TEST_PARTITION)


def copy_images_and_csvs(i, image_paths, permutation, dst_path):
    stem = image_paths[permutation[i]].stem
    name = image_paths[permutation[i]].name
    parent = image_paths[permutation[i]].parents[0]
    csv_name = Path(stem + ".csv")
    shutil.copyfile(image_paths[permutation[i]], dst_path / name)
    shutil.copyfile(parent / csv_name, dst_path / csv_name)


"""
python3 preprocessing-scripts/generate_dataset_images_directories.py ~/mac/Box/oct_segmentation_haohua_qian/WayneOCTimages_Sorted/ wayne-images/
"""
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python generate_dataset_images_directories.py </path/to/input/dir> </path/to/output/dir>")
        exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    image_paths = []
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if (file.endswith(".tiff") or file.endswith(".TIFF")) and not file.startswith("."):
                image_paths.append(Path(os.path.join(subdir, file)))

    print(f"Found {len(image_paths)} images")

    # Cleanup
    training_path = Path(output_dir + "/training")
    validation_path = Path(output_dir + "/validation")
    test_path = Path(output_dir + "/test")

    shutil.rmtree(training_path, ignore_errors=True)
    shutil.rmtree(validation_path, ignore_errors=True)
    shutil.rmtree(test_path, ignore_errors=True)

    os.makedirs(training_path)
    os.makedirs(validation_path)
    os.makedirs(test_path)
    
    permutation = np.random.permutation(len(image_paths))

    test_images = int(round(TEST_PARTITION * len(image_paths)))
    validation_images = int(round(VALIDATION_PARTITION * len(image_paths)))

    for i in range(test_images):
        copy_images_and_csvs(i, image_paths, permutation, test_path)

    for i in range(test_images, test_images + validation_images):
        copy_images_and_csvs(i, image_paths, permutation, validation_path)

    for i in range(test_images + validation_images, len(image_paths)):
        copy_images_and_csvs(i, image_paths, permutation, training_path)
