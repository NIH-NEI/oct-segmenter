import os
import sys

import logging as log
import numpy as np
from pathlib import Path

EXPECTED_NUMBER_OF_CLASSES = 7


def check_class_order_in_column(filename, mask, column):
    # Check layers reach left side of the image
    _, idx = np.unique(mask[:, column], return_index=True)
    ordered_classes = mask[np.sort(idx), column]
    if not np.array_equal(
        ordered_classes, np.arange(EXPECTED_NUMBER_OF_CLASSES)
    ):
        log.warning(
            (
                f"File name: {filename}. Column: {column}. Classes found: "
                f"{ordered_classes}"
            )
        )


if __name__ == "__main__":
    """
    This script takes as input a directory containing the mask files (in CSV \
    format) and checks that the all the expected retina layers are present \
    and in order.

    Example:
    python preprocessing-scripts/custom/check_retina_layers_order.py \
        data/experiment-10/images/
    """
    log.basicConfig()
    log.getLogger().setLevel(log.INFO)

    input_dir = Path(sys.argv[1])

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".csv") and not filename.startswith("."):
                mask = np.loadtxt(
                    Path(root) / Path(filename), delimiter=",", dtype=int
                )
                for column in range(mask.shape[1]):
                    check_class_order_in_column(filename, mask, column)
