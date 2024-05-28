import os
import shutil
import sys

import logging
import numpy as np
from pathlib import Path


def copy_json(i, image_paths, permutation, dst_path):
    name = image_paths[permutation[i]].name
    shutil.copyfile(image_paths[permutation[i]], dst_path / name)


def copy_images_and_csvs(i, image_paths, permutation, dst_path):
    stem = image_paths[permutation[i]].stem
    name = image_paths[permutation[i]].name
    parent = image_paths[permutation[i]].parents[0]
    csv_name = Path(stem + ".csv")
    shutil.copyfile(image_paths[permutation[i]], dst_path / name)
    shutil.copyfile(parent / csv_name, dst_path / csv_name)


def partition(args):
    """
    Given an input directory this command:
        1. Recursively finds all the .tiff images
        2. Creates a random permutation of [0 - # images found]
        3. Partitions the images into the training, validation and test datasets
    """

    # Check partitions add up to 1
    training_partition = args.training
    validation_partition = args.validation
    test_partition = args.test
    partition_sum = training_partition + validation_partition + test_partition

    if partition_sum != 1.0:
        print(f"Parititons sum is {partition_sum}. They should add up to 1")
        exit(1)

    logging.info(
        f"Paritioning images with the following proportions: training {training_partition}"
        f", validation: {validation_partition}, testing: {test_partition}"
    )

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print("Input directory doesn't exist")
        exit(1)

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        print("Output directory doesn't exist")
        exit(1)

    extension = ".tiff"
    if args.j:
        extension = ".json"

    image_paths = []
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if (
                file.endswith(extension) or file.endswith(extension.upper())
            ) and not file.startswith("."):
                image_paths.append(Path(os.path.join(subdir, file)))

    logging.info(f"Found {len(image_paths)} images")

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

    test_images = int(round(test_partition * len(image_paths)))
    validation_images = int(round(validation_partition * len(image_paths)))
    training_images = len(image_paths) - validation_images - test_images

    logging.info(f"Training Images: {training_images}")
    logging.info(f"Validation Images: {validation_images}")
    logging.info(f"Test Images: {test_images}")

    for i in range(test_images):
        if args.j:
            copy_json(i, image_paths, permutation, test_path)
        else:
            copy_images_and_csvs(i, image_paths, permutation, test_path)

    for i in range(test_images, test_images + validation_images):
        if args.j:
            copy_json(i, image_paths, permutation, validation_path)
        else:
            copy_images_and_csvs(i, image_paths, permutation, validation_path)

    for i in range(test_images + validation_images, len(image_paths)):
        if args.j:
            copy_json(i, image_paths, permutation, training_path)
        else:
            copy_images_and_csvs(i, image_paths, permutation, training_path)
