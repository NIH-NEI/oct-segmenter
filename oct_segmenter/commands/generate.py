from pathlib import Path

from oct_segmenter.preprocessing import test_dataset, training_dataset


def generate_training_dataset(args):
    dataset = training_dataset.generate_training_dataset(
        train_input_dir=Path(args.training_input_dir),
        validation_input_dir=Path(args.validation_input_dir),
        output_file=Path(args.output_dir) / Path("training_dataset.hdf5"),
        wayne_format=args.wayne_state_format,
    )
    dataset.close()


def generate_test_dataset(args):
    dataset = test_dataset.generate_test_dataset(
        test_input_dir=Path(args.test_input_dir),
        output_file=Path(args.output_dir) / Path("test_dataset.hdf5"),
        wayne_format=args.wayne_state_format,
    )

    dataset.close()
