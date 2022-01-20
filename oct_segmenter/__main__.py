import os

import argparse
import art
import logging

from oct_segmenter import DEFAULT_MODEL_INDEX, DEFAULT_TEST_PARTITION, DEFAULT_TRAINING_PARTITION,\
    DEFAULT_TEST_PARTITION, DEFAULT_VALIDATION_PARTITION
from oct_segmenter.commands.generate import generate_training_dataset, generate_test_dataset
from oct_segmenter.commands.list import list_models
from oct_segmenter.commands.partition import partition
from oct_segmenter.commands.predict import predict
from oct_segmenter.commands.train import train

def main():
    print(art.text2art("oct-segmenter"))

    # Set logging
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    # Create args-parser
    parser = argparse.ArgumentParser()
    cmd_subparser = parser.add_subparsers(dest="command", required=True)
    generate = cmd_subparser.add_parser("generate")

    generate_subparser = generate.add_subparsers(dest="generate", required=True)

    # Generate test dataset
    gen_test_parser = generate_subparser.add_parser("test")
    gen_test_parser.add_argument(
        "-t",
        "--test-input-dir",
        help="path to the directory containing test images",
        required=True,
    )

    gen_test_parser.add_argument(
        "-w",
        "--wayne-state-format",
        default=False,
        action="store_true",
        help="Generate dataset using the Wayne state format (vs. visual function core format)",
    )

    gen_test_parser.add_argument(
        "-o",
        "--output-dir",
        help="name of the output name file",
        default=".",
    )

    # Generate training dataset
    gen_train_parser = generate_subparser.add_parser("training")
    gen_train_parser.add_argument(
        "--training-input-dir",
        "-t",
        help="path to the directory containing training images",
        required=True,
    )

    gen_train_parser.add_argument(
        "-v",
        "--validation-input-dir",
        help="path to the directory containing validation images",
        required=True,
    )

    gen_train_parser.add_argument(
        "-w",
        "--wayne-state-format",
        default=False,
        action="store_true",
        help="Generate dataset using the Wayne state format (vs. visual function core format)",
    )

    gen_train_parser.add_argument(
        "-o",
        "--output-dir",
        help="name of the output name file",
        default="."
    )

    # Train
    train_subparser = cmd_subparser.add_parser("train")
    train_subparser.add_argument(
        "-i",
        "--input",
        help="input training dataset (hdf5 file)",
        required=True,
    )
    train_subparser.add_argument(
        "-o",
        "--output-dir",
        help="name of the output directory to save model",
        required=True,
    )

    # Partition
    partition_subparser = cmd_subparser.add_parser(
        "partition",
        help="Given an input directory, partition the images into the training, validation and test datasets"
    )

    partition_subparser.add_argument(
        "-i",
        "--input-dir",
        help="input directory with images and csvs",
        required=True,
    )

    partition_subparser.add_argument(
        "-o",
        "--output-dir",
        help="name of the output directory to save the training, validation and test images",
        required=True,
    )

    partition_subparser.add_argument(
        "--training",
        help="Fraction of the total images to use for the training dataset",
        type=float,
        default=DEFAULT_TRAINING_PARTITION
    )

    partition_subparser.add_argument(
        "--validation",
        help="Fraction of the total images to use for the validation dataset",
        type=float,
        default=DEFAULT_VALIDATION_PARTITION
    )

    partition_subparser.add_argument(
        "--test",
        help="Fraction of the total images to use for the test dataset",
        type=float,
        default=DEFAULT_TEST_PARTITION
    )

    # Predict
    predict_subparser = cmd_subparser.add_parser("predict")
    predict_input_group = predict_subparser.add_mutually_exclusive_group(required=True)
    predict_input_group.add_argument("--input", "-i", help="input file image to segment")
    predict_input_group.add_argument(
        "--input-dir", "-d", help="input directory containing .tiff images to be segmented."
    )

    predict_model_group = predict_subparser.add_mutually_exclusive_group(required=False)

    predict_model_group.add_argument(
        "--model-index",
        "-n",
        help="Model to use for prediction. Run 'oct-segmenter list' to see full list.",
        default=DEFAULT_MODEL_INDEX,
        type=int
    )

    predict_model_group.add_argument(
        "--model-path",
        "-m",
        help="Path to model to use for prediction (HDF5 file).",
        type=str,
    )

    predict_subparser.add_argument("-c",
        default=False,
        action="store_true",
        help="label complete PNG image instead of left/right regions"
    )

    predict_subparser.add_argument(
        "--label-png",
        "-l",
        help="output segmentation map PNG file"
    )

    predict_subparser.add_argument(
        "--output",
        "-o",
        help="output file or directory (if it ends with .csv it is "
        "recognized as file, else as directory)",
    )

    # List Models
    list_subparser = cmd_subparser.add_parser("list")

    args = parser.parse_args()

    if args.command == "generate":
        if args.generate == "test":
            generate_test_dataset(args)
        elif args.generate == "training":
            generate_training_dataset(args)
        else:
            print(
                "Unrecognized 'generate' option. Type: oct-segmenter \
                generate -h for help"
            )
            exit(1)

    elif args.command == "partition":
        partition(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "list":
        list_models()
    elif args.command == "train":
        train(args)


if __name__ == "__main__":
    main()
