import json
import logging as log
from pathlib import Path
from tensorflow.keras import optimizers

from oct_image_segmentation_models.common.mlflow_parameters import MLflowParameters
from oct_image_segmentation_models.common import augmentation as aug
from oct_image_segmentation_models.training import training
from oct_image_segmentation_models.training.training_parameters import TrainingParams

DEFAULT_AUGMENTATION_MODE = "none"
DEFAULT_BATCH_SIZE = 2
DEFAULT_EARLY_STOPPING = True
DEFAULT_EPOCHS = 1000
DEFAULT_LOSS = "dice_loss"
DEFAULT_METRIC = "dice_coef"
DEFAULT_PATIENCE = 50
DEFAULT_RESTORE_BEST_WEIGHTS = True
DEFAULT_CLASS_WEIGHT = None

DEFAULT_MLFLOW_EXPERIMENT_NAME = "mice-image-segmentation"
DEFAULT_MLFLOW_TRACKING_URI = Path.home() / Path("mlruns")
DEFAULT_MLFLOW_TRACKING_USERNAME = None
DEFUALT_MLFLOW_TRACKING_PASSWORD = None


def train(args):
    loss = DEFAULT_LOSS
    metric = DEFAULT_METRIC
    class_weight = DEFAULT_CLASS_WEIGHT
    epochs = DEFAULT_EPOCHS
    batch_size = DEFAULT_BATCH_SIZE
    aug_mode = DEFAULT_AUGMENTATION_MODE
    early_stopping = DEFAULT_EARLY_STOPPING
    mlflow_experiment_name = DEFAULT_MLFLOW_EXPERIMENT_NAME
    mlflow_tracking_uri = DEFAULT_MLFLOW_TRACKING_URI
    mlflow_tracking_username = DEFAULT_MLFLOW_TRACKING_USERNAME
    mlflow_tracking_password = DEFUALT_MLFLOW_TRACKING_PASSWORD
    patience = DEFAULT_PATIENCE
    restore_best_weights = DEFAULT_RESTORE_BEST_WEIGHTS

    if args.config:
        with open(args.config, "r") as f:
            config_data = json.load(f)
            batch_size = config_data.get("batch_size", DEFAULT_BATCH_SIZE)
            class_weight = config_data.get(
                "class_weight", DEFAULT_CLASS_WEIGHT
            )
            early_stopping = config_data.get(
                "early_stopping", DEFAULT_EARLY_STOPPING
            )
            loss = config_data.get("loss", DEFAULT_LOSS)
            metric = config_data.get("metric", DEFAULT_METRIC)
            epochs = config_data.get("epochs", DEFAULT_EPOCHS)
            augment = config_data.get("augment")
            if augment:
                aug_mode = "all"
            mlflow_experiment_name = config_data.get(
                "experiment", DEFAULT_MLFLOW_EXPERIMENT_NAME
            )
            mlflow_tracking_uri = config_data.get(
                "tracking_uri", DEFAULT_MLFLOW_TRACKING_URI
            )
            mlflow_tracking_username = config_data.get(
                "username", DEFAULT_MLFLOW_TRACKING_USERNAME
            )
            mlflow_tracking_password = config_data.get(
                "password", DEFUALT_MLFLOW_TRACKING_PASSWORD
            )
            patience = config_data.get("patience", DEFAULT_PATIENCE)
            restore_best_weights = config_data.get(
                "restore_best_weights", DEFAULT_RESTORE_BEST_WEIGHTS
            )

    log.info(f"Training Parameter: Early Stopping: {early_stopping}")
    log.info(f"Training Parameter: Loss: {loss}")
    log.info(f"Training Parameter: Metric: {metric}")
    log.info(f"Training Parameter: Epochs: {epochs}")
    log.info(f"Training Parameter: Batch Size: {batch_size}")
    log.info(f"Training Parameter: Class Weight: {class_weight}")
    log.info(f"Training Parameter: Augmentation: {aug_mode}")
    log.info(f"Training Parameter: Patience: {patience}")
    log.info(
        f"Training Parameter: Restore Best Weights: {restore_best_weights}"
    )
    log.info(f"MLFlow Tracking URI: {mlflow_tracking_uri}")
    log.info(f"MLFlow Experiment Name: {mlflow_experiment_name}")

    initial_model = Path(args.model) if args.model else None

    t_params = TrainingParams(
        training_dataset_path=Path(args.input).absolute(),
        initial_model=initial_model,
        results_location=Path(args.output_dir),
        opt_con=optimizers.Adam,
        opt_params={},
        loss=loss,
        metric=metric,
        epochs=epochs,
        batch_size=batch_size,
        aug_fn_args=[
            (aug.no_aug, {}),
            (aug.flip_aug, {"flip_type": "left-right"}),
        ],
        aug_mode=aug_mode,
        aug_probs=(0.5, 0.5),
        aug_val=False,
        aug_fly=True,
        model_save_best=True,
        class_weight=class_weight,
        early_stopping=early_stopping,
        restore_best_weights=restore_best_weights,
        patience=patience,
    )

    mlflow_params = MLflowParameters(
        mlflow_tracking_uri,
        username=mlflow_tracking_username,
        password=mlflow_tracking_password,
        experiment=mlflow_experiment_name,
    )

    training.train_model(
        t_params,
        mlflow_params,
    )
