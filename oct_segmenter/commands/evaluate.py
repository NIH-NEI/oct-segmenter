import os

import h5py
import logging as log
from pathlib import Path

from unet.common import dataset_loader as dl
from unet.evaluation import evaluation
from unet.evaluation.evaluation_parameters import (
    EvaluationParameters,
    EvaluationSaveParams,
    Dataset,
)

from oct_segmenter import MODELS_TABLE, MODELS_INDEX_MAP


def evaluate(args):

    if args.model_path:
        model_path = Path(args.model_path)
        model_name = model_path
    else:
        # Check selected model is valid
        if args.model_index is None:
            print(
                "oct-segementer: Looks like no model has been loaded. Make "
                "sure a model exists. Exiting..."
            )
            exit(1)

        number_of_models = len(MODELS_INDEX_MAP)
        if args.model_index >= number_of_models:
            print(
                f"Please select an index model from 0 to "
                f"{number_of_models - 1}. Exiting..."
            )
            exit(1)

        model_name = MODELS_INDEX_MAP[args.model_index]
        model_path = MODELS_TABLE[model_name]

    log.info(f"Using model: {model_name}")

    test_dataset_path = Path(args.input)

    if not test_dataset_path.is_file():
        print("oct-segmenter: Input file not found. Exiting...")
        exit(1)

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print("oct-segmenter: Output directory not found. Exiting...")
        exit(1)

    if any(output_dir.iterdir()):
        print("Output directory should be empty. Exiting...")
        exit(1)

    test_dataset_file = h5py.File(test_dataset_path, "r")

    test_images, test_labels, test_image_names = dl.load_testing_data(
        test_dataset_file
    )

    output_paths = []
    for i in range(test_images.shape[0]):
        output_paths.append(output_dir / Path(f"image_{i}"))

    test_dataset = Dataset(
        images=test_images,
        image_masks=test_labels,
        image_names=test_image_names,
        image_output_dirs=output_paths,
    )

    # Create output dirs
    for output_path in output_paths:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    save_params = EvaluationSaveParams(
        predicted_labels=True,
        categorical_pred=False,
        png_images=True,
        boundary_maps=True,
    )

    eval_params = EvaluationParameters(
        model_path=model_path,
        dataset=test_dataset,
        save_foldername=output_dir.absolute(),
        save_params=save_params,
        gsgrad=1,
        transpose=False,
        dice_errors=True,
        binarize=True,
        bg_ilm=True,
        bg_csi=False,
    )

    evaluation.evaluate_model(eval_params)
