import os
import shutil
import sys

import logging as log
from pathlib import Path

TEST_PARTITION = 0.3
TRAINING_PARTITION = round(0.8 * (1 - TEST_PARTITION), 2)
VALIDATION_PARTITION = round(0.2 * (1 - TEST_PARTITION), 2)


def copy_image_files(input_dir, output_dir, image_paths):
    shutil.rmtree(output_dir, ignore_errors=True)
    os.mkdir(output_dir)
    for image_file in image_paths:
        image_path = input_dir / Path(image_file)
        csv_file = Path(image_file[:-4] + "csv")
        csv_path = input_dir / csv_file
        shutil.copyfile(image_path, output_dir / Path(image_file))
        shutil.copyfile(csv_path, output_dir / csv_file)


if __name__ == "__main__":
    """
    This script takes as input a directory containing TIFF files and their \
    corresponding CSVs and a TSV file that maps image name to subject (see
    `preprocessing-scripts/custom/map_image_name_to_subject.py`). It splits \
    the images into the training, test and validation datasets making sure \
    that no subject appears in more than one partition.

    Example:
    python preprocessing-scripts/custom/split_images_into_train_val_test.py \
        data/experiment-10/filename_to_subject.tsv \
        data/experiment-10/images/
    """
    log.basicConfig()
    log.getLogger().setLevel(log.INFO)

    log.info(
        (
            f"Targets: Train: {TRAINING_PARTITION}, Test: {TEST_PARTITION}, "
            f"Validation: {VALIDATION_PARTITION}"
        )
    )
    filename_to_subject_filepath = Path(sys.argv[1])
    input_dir = Path(sys.argv[2])

    subject_to_filenames = {}
    total_images = 0

    with open(filename_to_subject_filepath, "r") as fts_file:
        for line in fts_file.readlines():
            filename, subject = line.split("\t")
            filenames = subject_to_filenames.get(subject, [])
            filenames.append(filename)
            subject_to_filenames[subject] = filenames
            total_images += 1

    log.info(f"Found {total_images} images.")
    test_images = round(total_images * TEST_PARTITION)
    train_images = round(total_images * TRAINING_PARTITION)
    validation_images = round(total_images * VALIDATION_PARTITION)

    test_image_paths = []
    train_image_paths = []
    validation_image_paths = []

    subject_to_image_count = dict(
        (subject, len(filenames))
        for (subject, filenames) in subject_to_filenames.items()
    )

    for subject, image_count in sorted(
        subject_to_image_count.items(), key=lambda item: item[1]
    ):
        if len(train_image_paths) < train_images:
            train_image_paths.extend(subject_to_filenames[subject])
            continue

        if len(test_image_paths) < test_images:
            test_image_paths.extend(subject_to_filenames[subject])
            continue

        validation_image_paths.extend(subject_to_filenames[subject])

    train_images_count = len(train_image_paths)
    test_images_count = len(test_image_paths)
    validation_images_count = len(validation_image_paths)
    log.info(
        f"The training dataset has {train_images_count}. "
        f"({round(train_images_count/total_images, 2)})"
    )
    log.info(
        f"The test dataset has {test_images_count}. "
        f"({round(test_images_count/total_images, 2)})"
    )
    log.info(
        f"The validation dataset has {validation_images_count}. "
        f"({round(validation_images_count/total_images, 2)})"
    )

    log.info(
        f"Total classifed images: "
        f"{train_images_count + test_images_count + validation_images_count}"
    )

    training_dir = input_dir / Path("training")
    test_dir = input_dir / Path("test")
    validation_dir = input_dir / Path("validation")

    copy_image_files(input_dir, training_dir, train_image_paths)
    copy_image_files(input_dir, test_dir, test_image_paths)
    copy_image_files(input_dir, validation_dir, validation_image_paths)
