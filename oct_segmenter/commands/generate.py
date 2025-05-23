from __future__ import annotations

import logging as log
from pathlib import Path
from typing import List

from oct_segmenter import (
    VISUAL_FUNCTION_CORE_LAYER_NAMES,
    WAYNE_STATE_LAYER_NAMES,
)
from oct_segmenter.preprocessing import test_dataset, training_dataset


def format_flags_to_string(args) -> str:
    if args.visual_function_core_format:
        return "visual"
    elif args.wayne_state_format:
        return "wayne"
    elif args.labelme_format:
        return "labelme"
    elif args.mask_format:
        return "mask"
    else:
        log.error(
            "Input format option not found. Confirm that one of '-f', '-w', "
            "'-m' or '-l' flags was provided."
        )
        exit(1)


def get_layer_list_from_layer_format_flag(format_flag: str) -> List[str]:
    if format_flag == "visual-function-core":
        return VISUAL_FUNCTION_CORE_LAYER_NAMES
    elif format_flag == "wayne-state":
        return WAYNE_STATE_LAYER_NAMES
    else:
        log.error(f"Unrecognized layer format option: {format_flag}")
        exit(1)


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

    input_format = format_flags_to_string(args)
    layer_names = (
        get_layer_list_from_layer_format_flag(args.layers_format)
        if input_format == "labelme"
        else None
    )

    dataset = training_dataset.generate_training_dataset(
        train_input_dir=train_input_dir,
        validation_input_dir=validation_input_dir,
        output_file=output_dir / Path("training_dataset.hdf5"),
        input_format=input_format,
        rgb_format=args.rgb,
        layer_names=layer_names,
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

    input_format = format_flags_to_string(args)
    layer_names = (
        get_layer_list_from_layer_format_flag(args.layers_format)
        if input_format == "labelme"
        else None
    )

    dataset = test_dataset.generate_test_dataset(
        test_input_dir=test_input_dir,
        output_file=output_dir / Path("test_dataset.hdf5"),
        input_format=input_format,
        rgb_format=args.rgb,
        layer_names=layer_names,
    )

    dataset.close()
