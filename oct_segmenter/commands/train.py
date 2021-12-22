
from pathlib import Path
from tensorflow.keras import optimizers

from unet.model import augmentation as aug
from unet.model import custom_losses
from unet.model import custom_metrics
from unet.model import training
from unet.model.training_parameters import TrainingParams


def train(args):
    # TODO: Add a way that the params can be read from a config file
    t_params = TrainingParams(
        training_dataset_path=Path(args.input).absolute(),
        training_dataset_name=Path(args.input).stem,
        results_location=args.output_dir,
        opt_con=optimizers.Adam,
        opt_params = {},
        loss=custom_losses.dice_loss,
        metric=custom_metrics.dice_coef,
        epochs=1000,
        batch_size=2,
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
