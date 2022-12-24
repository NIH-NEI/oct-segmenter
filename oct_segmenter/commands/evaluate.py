import os

import json
import logging as log
from pathlib import Path

from oct_image_segmentation_models.evaluation import evaluation
from oct_image_segmentation_models.evaluation.evaluation_parameters import (
    EvaluationParameters,
    EvaluationSaveParams,
)

from oct_segmenter import (
    DEFAULT_MLFLOW_TRACKING_URI,
    MODELS_TABLE,
    MODELS_INDEX_MAP,
)

DEFAULT_GRAPH_SEARCH = True
DEFAULT_METRICS = ["dice"]


def evaluate(args):
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
                f"Please select an index model from 0 to "
                f"{number_of_models - 1}. Exiting..."
            )
            exit(1)

        model_name = MODELS_INDEX_MAP[args.model_index]
        model_path = MODELS_TABLE[model_name]

    log.info(f"Using model: {model_name}")

    test_dataset_path = Path(args.input)

    if not test_dataset_path.is_file():
        print("oct-segmenter: Test Dataset file not found. Exiting...")
        exit(1)

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print("oct-segmenter: Output directory not found. Exiting...")
        exit(1)

    if any(output_dir.iterdir()):
        print("Output directory should be empty. Exiting...")
        exit(1)

    graph_search = DEFAULT_GRAPH_SEARCH
    metrics = DEFAULT_METRICS
    if args.config:
        with open(args.config, "r") as f:
            config_data = json.load(f)
            graph_search = config_data.get(
                "graph_search",
                DEFAULT_GRAPH_SEARCH,
            )
            metrics = config_data.get(
                "metrics",
                DEFAULT_METRICS,
            )

    log.info(f"Evaluation Parameter: Graph Search: {graph_search}")
    log.info(f"Evaluation Parameter: Metrics: {metrics}")

    save_params = EvaluationSaveParams(
        predicted_labels=True,
        categorical_pred=False,
        png_images=True,
        boundary_maps=True,
    )

    eval_params = EvaluationParameters(
        model_path=model_path,
        mlflow_tracking_uri=mlflow_tracking_uri,
        test_dataset_path=test_dataset_path,
        save_foldername=output_dir.absolute(),
        save_params=save_params,
        graph_search=graph_search,
        metrics=metrics,
        gsgrad=1,
        dice_errors=True,
        binarize=True,
        bg_ilm=True,
        bg_csi=False,
    )

    evaluation.evaluate_model(eval_params)
