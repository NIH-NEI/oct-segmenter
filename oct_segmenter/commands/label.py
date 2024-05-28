import os

import logging as log
import json
import numpy as np
from pathlib import Path
from PIL import Image

from oct_segmenter.postprocessing.postprocessing import (
    create_labelme_file_from_boundaries,
)


def label(args):
    input_paths = []
    if args.input:
        input_path = Path(args.input)
        input_dir = input_path.parent
        if not input_path.is_file():
            print("oct-segmenter: Input file not found. Exiting...")
            exit(1)
        input_paths.append(input_path)
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print("oct-segmenter: Input directory not found. Exiting...")
            exit(1)

        for subdir, _, files in os.walk(args.input_dir):
            for file in files:
                if (
                    file.endswith(".tiff") or file.endswith(".TIFF")
                ) and not file.startswith("."):
                    input_paths.append(Path(os.path.join(subdir, file)))
    else:
        print(
            "oct-segmenter: No input image file or directory were provided. Exiting..."
        )
        exit(1)

    root_output_dir = Path(args.output_dir)
    if not root_output_dir.is_dir():
        print("oct-segmenter: Output directory not found. Exiting...")
        exit(1)

    for input_path in input_paths:
        log.info(f"Generating 'labelme' file for image: {input_path}")
        boundaries_path = Path(input_path.parent) / Path(input_path.stem + ".csv")
        if not boundaries_path.is_file():
            log.warn(f"Boundaries file '{boundaries_path}' not found. Skipping...")
            continue

        img_arr = np.array(Image.open(input_path))
        boundaries = np.genfromtxt(boundaries_path, delimiter=",")

        labelme_data = create_labelme_file_from_boundaries(
            img_arr, input_path, boundaries
        )
        if labelme_data:
            output_dir = root_output_dir / input_path.parent.relative_to(input_dir)
            with open(output_dir / Path(input_path.stem + ".json"), "w") as file:
                json.dump(labelme_data, file)
        else:
            log.warn(
                f"Encountered error when processing image: {input_path}. Skipping..."
            )
