import json
import logging as log
from pathlib import Path
from tensorflow.keras import optimizers

from unet.model import augmentation as aug
from unet.model import custom_losses
from unet.model import custom_metrics
from unet.model import training
from unet.model.training_parameters import TrainingParams

DEFAULT_EPOCHS = 1000
DEFAULT_BATCH_SIZE = 2

def train(args):

    epochs = DEFAULT_EPOCHS
    batch_size = DEFAULT_BATCH_SIZE

    if args.config:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
            batch_size = config_data.get("batch_size", DEFAULT_BATCH_SIZE)
            epochs = config_data.get("epochs", DEFAULT_EPOCHS)

    log.info(f"Training Parameter: Epochs: {epochs}")
    log.info(f"Training Parameter: Batch Size: {batch_size}")

    t_params = TrainingParams(
        training_dataset_path=Path(args.input).absolute(),
        training_dataset_name=Path(args.input).stem,
        results_location=args.output_dir,
        opt_con=optimizers.Adam,
        opt_params = {},
        loss=custom_losses.dice_loss,
        metric=custom_metrics.dice_coef,
        epochs=epochs,
        batch_size=batch_size,
        aug_fn_args=[(aug.no_aug, {}), (aug.flip_aug, {"flip_type": "left-right"})],
        aug_mode="one",
        aug_probs=(0.5, 0.5),
        aug_val=False,
        aug_fly=True,
        model_save_best=True,
    )

    training.train_model(
        t_params,
    )
