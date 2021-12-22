import os
import sys

import h5py
from pathlib import Path

from oct_segmenter.preprocessing import generic_dataset as generator


def generate_training_dataset(
    train_input_dir: Path,
    validation_input_dir: Path,
    output_file: Path,
    wayne_format: bool=False,
) -> h5py.File:
    training_dataset = generator.generate_generic_dataset(train_input_dir, output_file, wayne_format)
    validation_dataset = generator.generate_generic_dataset(
        validation_input_dir,
        Path("/tmp/validation.hdf5"),
        wayne_format,
        backing_store=False,
    )

    training_dataset["train_images"] = training_dataset["xhat"]
    training_dataset["train_labels"] = training_dataset["yhat"]
    training_dataset["train_images_source"] = training_dataset["image_source"]

    del training_dataset["xhat"]
    del training_dataset["yhat"]
    del training_dataset["image_source"]

    training_dataset.create_dataset("val_images", data=validation_dataset["xhat"])
    training_dataset.create_dataset("val_labels", data=validation_dataset["yhat"])
    training_dataset.create_dataset("val_images_source", data=validation_dataset["image_source"])

    return training_dataset
