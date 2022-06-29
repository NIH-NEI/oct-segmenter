import os

import argparse
import art
import logging as log

from oct_segmenter import DEFAULT_MODEL_INDEX, DEFAULT_TEST_PARTITION, DEFAULT_TRAINING_PARTITION,\
    DEFAULT_TEST_PARTITION, DEFAULT_VALIDATION_PARTITION
from oct_segmenter.commands.evaluate import evaluate
from oct_segmenter.commands.generate import generate_training_dataset, generate_test_dataset
from oct_segmenter.commands.list import list_models
from oct_segmenter.commands.partition import partition
from oct_segmenter.commands.predict import predict
from oct_segmenter.commands.train import train

def main():
    print(art.text2art("oct-segmenter"))

    # Set logging
    log.basicConfig()
    log.getLogger().setLevel(log.INFO)

    # Create args-parser
    parser = argparse.ArgumentParser()
    cmd_subparser = parser.add_subparsers(dest="command", required=True)
    generate = cmd_subparser.add_parser("generate")

    generate_subparser = generate.add_subparsers(dest="generate", required=True)

    # Generate test dataset
    gen_test_parser = generate_subparser.add_parser("test")
    gen_test_parser.add_argument(
        "-i",
        "--test-input-dir",
        help="Path to the directory containing test images",
        required=True,
    )

    gen_test_csv_format_group = gen_test_parser.add_mutually_exclusive_group(required=True)

    gen_test_csv_format_group.add_argument(
        "-f",
        "--visual-function-core-format",
        default=False,
        action="store_true",
        help="Generate dataset using the Wayne State University format. (.tiff + 3 layer .csv)",
    )

    gen_test_csv_format_group.add_argument(
        "-w",
        "--wayne-state-format",
        default=False,
        action="store_true",
        help="Generate dataset using the Wayne state format. (.tiff + 6 layer .csv)",
    )

    gen_test_csv_format_group.add_argument(
        "-l",
        "--labelme-format",
        default=False,
        action="store_true",
        help="Generate dataset using the 'labelme' format. ('labelme' compatible .json file)",
    )

    gen_test_csv_format_group.add_argument(
        "-m",
        "--mask-format",
        default=False,
        action="store_true",
        help="Generate dataset using the 'mask' format. (.tiff + matrix mask .csv file)",
    )

    gen_test_parser.add_argument(
        "--layers-format",
        choices=["visual-function-core", "wayne-state"],
        help="Required when using '-l' flag. Visual Function Core layers: ['ILM', 'ELM', 'RPE']. Wayne State Layers: ['RNFL-vitreous', 'GCL-RNFL', 'INL-IPL', 'ONL-OPL', 'ELM', 'RPE']",
    )

    gen_test_parser.add_argument(
        "-o",
        "--output-dir",
        help="Name of the output name file",
        default=".",
    )

    # Generate training dataset
    gen_train_parser = generate_subparser.add_parser("training")
    gen_train_parser.add_argument(
        "-i",
        "--training-input-dir",
        help="Path to the directory containing training images",
        required=True,
    )

    gen_train_parser.add_argument(
        "-v",
        "--validation-input-dir",
        help="Path to the directory containing validation images",
        required=True,
    )

    gen_train_csv_format_group = gen_train_parser.add_mutually_exclusive_group(required=True)

    gen_train_csv_format_group.add_argument(
        "-f",
        "--visual-function-core-format",
        default=False,
        action="store_true",
        help="Generate dataset using the Wayne State University format. (.tiff + 3 layer .csv)",
    )

    gen_train_csv_format_group.add_argument(
        "-w",
        "--wayne-state-format",
        default=False,
        action="store_true",
        help="Generate dataset using the Wayne State University format. (.tiff + 6 layer .csv)",
    )

    gen_train_csv_format_group.add_argument(
        "-l",
        "--labelme-format",
        default=False,
        action="store_true",
        help="Generate dataset using the 'labelme' format. ('labelme' compatible .json file)",
    )

    gen_train_csv_format_group.add_argument(
        "-m",
        "--mask-format",
        default=False,
        action="store_true",
        help="Generate dataset using the mask format. (.tiff + matrix mask .csv file)",
    )

    gen_train_parser.add_argument(
        "--layers-format",
        choices=["visual-function-core", "wayne-state"],
        help="Required when using '-l' flag. Visual Function Core layers: ['ILM', 'ELM', 'RPE']. Wayne State Layers: ['RNFL-vitreous', 'GCL-RNFL', 'INL-IPL', 'ONL-OPL', 'ELM', 'RPE']",
    )

    gen_train_parser.add_argument(
        "-o",
        "--output-dir",
        help="Name of the output name file",
        default="."
    )

    # Train
    train_subparser = cmd_subparser.add_parser("train")
    train_subparser.add_argument(
        "-i",
        "--input",
        help="Input training dataset (hdf5 file)",
        required=True,
    )
    train_subparser.add_argument(
        "-o",
        "--output-dir",
        help="Name of the output directory to save model",
        required=True,
    )

    train_subparser.add_argument(
        "-c",
        "--config",
        help="Path to JSON config file",
        required=False,
    )

    train_subparser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Path to original model for retraining",
        required=False,
    )

    # Partition
    partition_subparser = cmd_subparser.add_parser(
        "partition",
        help="Given an input directory, partition the images into the training, validation and test datasets"
    )

    partition_subparser.add_argument(
        "-j",
        default=False,
        action="store_true",
        help="json",
    )

    partition_subparser.add_argument(
        "-i",
        "--input-dir",
        help="Input directory with images and CSVs",
        required=True,
    )

    partition_subparser.add_argument(
        "-o",
        "--output-dir",
        help="Name of the output directory to save the training, validation and test images",
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
    predict_input_group.add_argument(
        "--input",
        "-i",
        help="input file image to segment"
    )
    predict_input_group.add_argument(
        "--input-dir",
        "-d",
        help="input directory containing .tiff images to be segmented."
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
        help="Label complete PNG image instead of left/right regions"
    )

    predict_subparser.add_argument(
        "--flip-top-bottom",
        "-f",
        default=False,
        action="store_true",
        help="Flip images w.r.t. the horizontal axis before prediction"
    )

    predict_subparser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory",
    )

    # Evaluate
    evaluate_subparser = cmd_subparser.add_parser("evaluate")

    evaluate_subparser.add_argument(
        "--input",
        "-i",
        required=True,
        help="input test dataset HDF5 file"
    )

    evaluate_model_group = evaluate_subparser.add_mutually_exclusive_group(required=False)
    evaluate_model_group.add_argument(
        "--model-index",
        "-n",
        default=DEFAULT_MODEL_INDEX,
        type=int,
        help="Model to use for evaluation. Run 'oct-segmenter list' to see full list.",
    )

    evaluate_model_group.add_argument(
        "--model-path",
        "-m",
        help="Path to model to use for evaluation (HDF5 file).",
        type=str,
    )

    evaluate_subparser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Output directory",
    )

    # List Models
    list_subparser = cmd_subparser.add_parser("list")

    args = parser.parse_args()

    if args.command == "generate":
        if args.labelme_format and args.layers_format is None:
            log.error("If generating images from 'labelme' files specify the layer format with the '--layers-format' flag")
            exit(1)
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
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "list":
        list_models()
    elif args.command == "train":
        train(args)


if __name__ == "__main__":
    main()
