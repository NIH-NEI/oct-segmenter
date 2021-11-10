import os

import numpy as np
from pathlib import Path

from oct_segmenter import __version__
from oct_segmenter.preprocessing import preprocess
from unet.model import eval_model

MODEL = os.path.dirname(os.path.abspath(__file__)) + "/../data/model/" + __version__ + "/model.hdf5"


def predict(args):
    input_paths = []
    if args.input:
        path = Path(args.input)
        if not path.is_file():
            print("oct-segmenter: Input file not found. Exiting...")
            exit(1)
        input_paths.append(path)
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print("oct-segmenter: Input directory not found. Exiting...")
            exit(1)

        for subdir, _, files in os.walk(args.input_dir):
            for file in files:
                if file.endswith(".tiff") or file.endswith(".TIFF"):
                    input_paths.append(Path(os.path.join(subdir, file)))
    else:
        print("oct-segmenter: No input image file or directory were provided. Exiting...")
        exit(1)

    for input_path in input_paths:
        if args.output:
            output = Path(args.output)
        else:
            output = input_path.parent

        if args.c:
            img = preprocess.generate_input_image(input_path)
            pred_images = np.array([img])
            pred_image_names = [Path(input_path.stem + "_labeled" + input_path.suffix)]
        else:
            img_left, img_right = preprocess.generate_side_region_input_image(input_path)
            pred_images = np.array([img_left, img_right])

            img_left_path = Path(input_path.stem + "_left" + input_path.suffix)
            img_right_path = Path(input_path.stem + "_right" + input_path.suffix)
            pred_image_names = [img_left_path, img_right_path]

        eval_model.evaluate_model(
            MODEL,
            pred_images,
            pred_image_names,
            False,
            output
        )
