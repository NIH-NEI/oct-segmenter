import os

import logging as log
import numpy as np
from pathlib import Path

from unet.model import augmentation as aug
from unet.model import evaluation
from unet.model.evaluation_parameters import EvaluationParameters, PredictionDataset
from unet.model.save_parameters import SaveParameters

from oct_segmenter import MODELS_TABLE, MODELS_INDEX_MAP
from oct_segmenter.preprocessing import preprocess


def predict(args):

    if args.model_path:
        model_path = Path(args.model_path)
        model_name = model_path
    else:
        # Check selected model is valid
        if args.model_index == None:
            print("oct-segementer: Looks like no model has been loaded. Make sure a model exists. Exiting...")
            exit(1)

        number_of_models = len(MODELS_INDEX_MAP)
        if args.model_index >= number_of_models:
            print(f"Please select an index model from 0 to {number_of_models - 1}. Exiting...")
            exit(1)

        model_name = MODELS_INDEX_MAP[args.model_index]
        model_path = MODELS_TABLE[model_name]
    
    log.info(f"Using model: {model_name}")

    input_paths = []
    if args.input:
        input_path = Path(args.input)
        input_dir = input_path.parent
        if not input_path.is_file():
            print("oct-segmenter: Input file not found. Exiting...")
            exit(1)
        input_paths.append(input_path)
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print("oct-segmenter: Input directory not found. Exiting...")
            exit(1)

        for subdir, _, files in os.walk(args.input_dir):
            for file in files:
                if (file.endswith(".tiff") or file.endswith(".TIFF")) and not file.startswith("."):
                    input_paths.append(Path(os.path.join(subdir, file)))
    else:
        print("oct-segmenter: No input image file or directory were provided. Exiting...")
        exit(1)

    if args.output_dir:
        root_output_dir = Path(args.output_dir)
    else:
        root_output_dir = input_dir

    pred_images = []
    pred_images_names = []
    output_paths = []
    for input_path in input_paths:
        if args.output_dir:
            output = root_output_dir / input_path.parent.relative_to(input_dir)
        else:
            output = input_path.parent

        if args.c:
            img = preprocess.generate_input_image(input_path, args.flip_top_bottom)
            pred_images.append(img)
            pred_images_names.append(Path(input_path.stem + "_labeled" + input_path.suffix))
            output_paths.append(output)
        else:
            img_left, img_right = preprocess.generate_side_region_input_image(
                input_path,
                args.flip_top_bottom
            )
            pred_images.append(img_left)
            pred_images.append(img_right)
            img_left_path = Path(input_path.stem + "_left" + input_path.suffix)
            pred_images_names.append(img_left_path)
            img_right_path = Path(input_path.stem + "_right" + input_path.suffix)
            pred_images_names.append(img_right_path)
            output_paths.extend([output, output])

    pred_images = np.array(pred_images)
    pred_dataset = PredictionDataset(
        pred_images,
        pred_images_names,
        output_paths,
    )

    # Create output dirs
    for output_path in output_paths:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    save_params = SaveParameters(
        pngimages=True,
        raw_image=True,
        raw_labels=True,
        temp_extra=True,
        boundary_maps=True,
        area_maps=False,
        comb_area_maps=True,
        seg_plot=True
    )

    eval_params = EvaluationParameters(
        model_file_path=model_path,
        prediction_dataset=pred_dataset,
        is_evaluate=False,
        col_error_range=None,
        save_foldername=root_output_dir,
        eval_mode="both",
        aug_fn_arg=(aug.no_aug, {}),
        save_params=save_params,
        verbosity=3,
        gsgrad=1,
        transpose=False,
        normalise_input=True,
        comb_pred=False,
        recalc_errors=False,
        boundaries=True,
        boundary_errors=False,
        trim_maps=False,
        trim_ref_ind=0,
        trim_window=(0, 0),
        dice_errors=True,
        flatten_image=False,
        flatten_ind=0,
        flatten_poly=False,
        binarize=True,
        binarize_after=True,
        bg_ilm=True,
        bg_csi=False,
        flatten_pred_edges=False,
        flat_marg=0,
        use_thresh=False,
        thresh=0.5,
    )

    evaluation.evaluate_model(eval_params)
