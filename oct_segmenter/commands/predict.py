import os

import json
import logging as log
import numpy as np
from pathlib import Path

from oct_image_segmentation_models.common.dataset import Dataset
from oct_image_segmentation_models.prediction import prediction
from oct_image_segmentation_models.prediction.prediction_parameters import (
    PredictionParams,
    PredictionSaveParams,
)

from oct_segmenter import (
    DEFAULT_MLFLOW_TRACKING_URI,
    MODELS_TABLE,
    MODELS_INDEX_MAP,
)
from oct_segmenter.preprocessing import preprocess
from oct_segmenter.postprocessing.postprocessing import (
    create_labelme_file_from_boundaries,
)


def predict(args):
    mlflow_tracking_uri = DEFAULT_MLFLOW_TRACKING_URI

    if args.model_path:
        model_path = Path(args.model_path)
        model_name = model_path
    elif args.mlflow_run_uuid:
        model_path = Path(f"runs:/{args.mlflow_run_uuid}/model")
        mlflow_tracking_uri = Path.home() / Path("mlruns")
        if os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow_tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        model_name = str(mlflow_tracking_uri) + "/" + str(model_path)
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
                "Please select an index model from 0 to "
                f"{number_of_models - 1}. Exiting..."
            )
            exit(1)

        model_name = MODELS_INDEX_MAP[args.model_index]
        model_path = MODELS_TABLE[model_name]

    log.info(f"Using model: {model_name}")

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
            "oct-segmenter: No input image file or directory were provided. "
            "Exiting..."
        )
        exit(1)

    if args.output_dir:
        root_output_dir = Path(args.output_dir)
        if not root_output_dir.is_dir():
            print("oct-segmenter: Output directory not found. Exiting...")
            exit(1)
    else:
        root_output_dir = input_dir

    pred_images = []
    pred_images_names = []
    output_paths = []
    for input_path in input_paths:
        if args.output_dir:
            output = root_output_dir / input_path.parent.relative_to(input_dir)
        else:
            output = input_path.parent

        if args.c:
            img = preprocess.generate_input_image(
                input_path, args.flip_top_bottom
            )
            if img is not None:
                pred_images.append(img)
                pred_images_names.append(Path(input_path.name))
                output_paths.append(
                    output / Path(input_path.stem + "_labeled")
                )
        else:
            img_left, img_right = preprocess.generate_side_region_input_image(
                input_path, args.flip_top_bottom
            )
            if img_left is not None:
                pred_images.append(img_left)
                pred_images.append(img_right)
                img_left_path = Path(
                    input_path.stem + "_left" + input_path.suffix
                )
                pred_images_names.append(img_left_path)
                img_right_path = Path(
                    input_path.stem + "_right" + input_path.suffix
                )
                pred_images_names.append(img_right_path)
                output_paths.extend(
                    [
                        output / Path(input_path.stem + "_left"),
                        output / Path(input_path.stem + "_right"),
                    ]
                )

    if len(pred_images) == 0:
        log.info("No images were processed successfully. Exiting...")
        exit(1)

    pred_images = np.array(pred_images)
    dataset = Dataset(
        images=pred_images,
        image_masks=None,
        image_names=pred_images_names,
        image_output_dirs=output_paths,
    )

    save_params = PredictionSaveParams(
        predicted_labels=True,
        categorical_pred=False,
        png_images=True,
        boundary_maps=True,
    )

    predict_params = PredictionParams(
        model_path=model_path,
        mlflow_tracking_uri=mlflow_tracking_uri,
        dataset=dataset,
        config_output_dir=root_output_dir,
        save_params=save_params,
        flatten_image=False,
        flatten_ind=0,
        flatten_poly=False,
        flatten_pred_edges=False,
        flat_marg=0,
        trim_maps=False,
        trim_ref_ind=0,
        trim_window=(0, 0),
        col_error_range=None,
    )

    # Create output dirs
    for output_path in output_paths:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    prediction_outputs = prediction.predict(predict_params)

    for prediction_output in prediction_outputs:
        image_name = prediction_output.image_name
        labelme_data = create_labelme_file_from_boundaries(
            # Image is in "batch" format and transposed for model processing.
            # Reverting it.
            img_arr=np.transpose(np.squeeze(prediction_output.image)),
            image_name=image_name,
            boundaries=prediction_output.delineations,
        )

        with open(
            prediction_output.image_output_dir
            / Path(image_name.stem + ".json"),
            "w",
        ) as file:
            json.dump(labelme_data, file)
