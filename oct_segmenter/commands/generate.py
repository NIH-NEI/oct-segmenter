from pathlib import Path

from oct_segmenter.preprocessing import test_dataset, training_dataset


def generate_training_dataset(args):
    train_input_dir = Path(args.training_input_dir)
    if not train_input_dir.is_dir():
        print("oct-segmenter: Training input directory not found. Exiting...")
        exit(1)

    validation_input_dir = Path(args.validation_input_dir)
    if not validation_input_dir.is_dir():
        print("oct-segmenter: Validation input directory not found. Exiting...")
        exit(1)

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print("oct-segmenter: Output directory not found. Exiting...")
        exit(1)

    dataset = training_dataset.generate_training_dataset(
        train_input_dir=train_input_dir,
        validation_input_dir=validation_input_dir,
        output_file=output_dir / Path("training_dataset.hdf5"),
        wayne_format=args.wayne_state_format,
    )
    dataset.close()


def generate_test_dataset(args):
    test_input_dir = Path(args.test_input_dir)
    if not test_input_dir.is_dir():
        print("oct-segmenter: Test input directory not found. Exiting...")
        exit(1)

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print("oct-segmenter: Output directory not found. Exiting...")
        exit(1)

    dataset = test_dataset.generate_test_dataset(
        test_input_dir=test_input_dir,
        output_file=output_dir / Path("test_dataset.hdf5"),
        wayne_format=args.wayne_state_format,
    )

    dataset.close()
