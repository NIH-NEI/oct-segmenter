import json
import logging as log
from pathlib import Path
from tensorflow.keras import optimizers

from unet.common.mlflow_parameters import MLflowParameters
from unet.model import augmentation as aug
from unet.model import custom_losses
from unet.model import custom_metrics
from unet.model import training
from unet.model.training_parameters import TrainingParams

DEFAULT_AUGMENTATION_MODE = "none"
DEFAULT_BATCH_SIZE = 2
DEFAULT_EPOCHS = 1000


def train(args):
    epochs = DEFAULT_EPOCHS
    batch_size = DEFAULT_BATCH_SIZE
    aug_mode = DEFAULT_AUGMENTATION_MODE

    if args.config:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
            batch_size = config_data.get("batch_size", DEFAULT_BATCH_SIZE)
            epochs = config_data.get("epochs", DEFAULT_EPOCHS)
            augment = config_data.get("augment")
            if augment:
                aug_mode = "all"
            mlflow_experiment_name = config_data.get("experiment")
            mlflow_tracking_uri = config_data.get("tracking_uri")
            mlflow_tracking_username = config_data.get("username")
            mlflow_tracking_password = config_data.get("password")

    log.info(f"Training Parameter: Epochs: {epochs}")
    log.info(f"Training Parameter: Batch Size: {batch_size}")
    log.info(f"Training Parameter: Augmentation: {aug_mode}")

    initial_model = Path(args.model) if args.model else None

    t_params = TrainingParams(
        training_dataset_path=Path(args.input).absolute(),
        training_dataset_name=Path(args.input).stem,
        initial_model=initial_model,
        results_location=args.output_dir,
        opt_con=optimizers.Adam,
        opt_params = {},
        loss=custom_losses.dice_loss,
        metric=custom_metrics.dice_coef,
        epochs=epochs,
        batch_size=batch_size,
        aug_fn_args=[(aug.no_aug, {}), (aug.flip_aug, {"flip_type": "left-right"})],
        aug_mode=aug_mode,
        aug_probs=(0.5, 0.5),
        aug_val=False,
        aug_fly=True,
        model_save_best=True,
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
