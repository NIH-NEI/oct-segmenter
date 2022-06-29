import h5py
from pathlib import Path
from typeguard import typechecked

from oct_segmenter.preprocessing import generic_dataset as generator


@typechecked
def generate_training_dataset(
    train_input_dir: Path,
    validation_input_dir: Path,
    output_file: Path,
    input_format: str,
    layer_names: list[str],
) -> h5py.File:
    training_dataset = generator.generate_generic_dataset(
        train_input_dir,
        output_file,
        input_format,
        layer_names,
    )
    validation_dataset = generator.generate_generic_dataset(
        validation_input_dir,
        Path("/tmp/validation.hdf5"),
        input_format,
        layer_names,
        backing_store=False,
    )

    training_dataset["train_images"] = training_dataset["xhat"]
    training_dataset["train_labels"] = training_dataset["yhat"]
    training_dataset["train_segs"] = training_dataset["segs"]
    training_dataset["train_images_source"] = training_dataset["image_source"]

    del training_dataset["xhat"]
    del training_dataset["yhat"]
    del training_dataset["segs"]
    del training_dataset["image_source"]

    training_dataset.create_dataset("val_images", data=validation_dataset["xhat"])
    training_dataset.create_dataset("val_labels", data=validation_dataset["yhat"])
    training_dataset.create_dataset("val_segs", data=validation_dataset["segs"])
    training_dataset.create_dataset("val_images_source", data=validation_dataset["image_source"])

    return training_dataset
