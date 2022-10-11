import os
import sys

import logging as log
import numpy as np
from pathlib import Path

EXPECTED_NUMBER_OF_CLASSES = 7


def calculate_classes_fraction(image: np.array):
    _, counts = np.unique(image, return_counts=True)

    total_count = np.sum(counts)
    return counts / total_count


if __name__ == "__main__":
    """
    This script takes as input a directory containing the mask files (in CSV \
    format) and calculates the class fractions of each image. It then prints \
    the average class fractions across all the images.

    Example:
    python preprocessing-scripts/trimming/calculate_classes_fraction.py \
        data/experiment-12/images/
    """
    log.basicConfig()
    log.getLogger().setLevel(log.INFO)

    image_class_fractions = np.empty((0, EXPECTED_NUMBER_OF_CLASSES))

    input_dir = Path(sys.argv[1])

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".csv") and not filename.startswith("."):
                mask = np.loadtxt(
                    Path(root) / Path(filename), delimiter=",", dtype=int
                )
                image_class_fractions = np.vstack(
                    [image_class_fractions, calculate_classes_fraction(mask)]
                )

    print("Average Class Fractions")
    for i in range(image_class_fractions.shape[1]):
        print(f"Class {i}: {np.average(image_class_fractions[:, i]):.2f}")
