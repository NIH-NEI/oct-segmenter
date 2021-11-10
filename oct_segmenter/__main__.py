import argparse

import art

from oct_segmenter.commands.predict import predict
from oct_segmenter.preprocessing import generate_test_dataset, generate_training_dataset

def main():
    print(art.text2art("oct-segmenter"))
    parser = argparse.ArgumentParser()
    cmd_subparser = parser.add_subparsers(dest="command", required=True)
    generate = cmd_subparser.add_parser("generate")

    generate_subparser = generate.add_subparsers(dest="generate", required=True)

    # Generate test dataset
    test_parser = generate_subparser.add_parser("test")
    test_parser.add_argument(
        "--test-input-dir", "-t", help="path to the directory containing test images", required=True
    )

    test_parser.add_argument("--output", "-o", help="name of the output name file")

    # Generate training dataset
    train_parser = generate_subparser.add_parser("train")
    train_parser.add_argument(
        "--train-input-dir",
        "-t",
        help="path to the directory containing training images",
        required=True,
    )

    train_parser.add_argument(
        "--validation-input-dir", "-v", help="path to the directory containing validation images"
    )

    train_parser.add_argument("--output", "-o", help="name of the output name file")

    # Predict
    predict_subparser = cmd_subparser.add_parser("predict")
    predict_input_group = predict_subparser.add_mutually_exclusive_group(required=True)
    predict_input_group.add_argument("--input", "-i", help="input file image to segment")
    predict_input_group.add_argument(
        "--input-dir", "-d", help="input directory containing .tiff images to be segmented."
    )

    predict_subparser.add_argument("-c", default=False, action="store_true",
        help="label complete PNG image instead of left/right regions")

    predict_subparser.add_argument("--label-png", "-l", help="output segmentation map PNG file")

    predict_subparser.add_argument(
        "--output",
        "-O",
        "-o",
        help="output file or directory (if it ends with .csv it is "
        "recognized as file, else as directory)",
    )

    args = parser.parse_args()

    if args.command == "generate":
        output = args.output

        if args.generate == "test":
            if output is None:
                output = "test_dataset.hdf5"
            generate_test_dataset.generate_test_dataset(args.test_input_dir, output)
        elif args.generate == "train":
            generate_training_dataset.generate_datasets()
        else:
            print(
                "Unrecognized 'generate' option. Type: oct-segmenter \
                generate -h for help"
            )
            exit(1)

    elif args.command == "predict":
        predict(args)


if __name__ == "__main__":
    main()
