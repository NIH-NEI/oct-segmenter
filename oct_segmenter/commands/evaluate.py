import os

import h5py
import logging as log
import numpy as np
from pathlib import Path

from unet.model import augmentation as aug
from unet.model import dataset_loader as dl
from unet.model import evaluation
from unet.model.evaluation_parameters import EvaluationParameters, Dataset
from unet.model.save_parameters import SaveParameters

from oct_segmenter import MODELS_TABLE, MODELS_INDEX_MAP


def evaluate(args):

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

    test_dataset_path = Path(args.input)

    if not test_dataset_path.is_file():
        print("oct-segmenter: Input file not found. Exiting...")
        exit(1)

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print("oct-segmenter: Output directory not found. Exiting...")
        exit(1)

    test_dataset_file = h5py.File(test_dataset_path, 'r')

    test_images, test_labels, test_segments, test_image_names = dl.load_testing_data(
        test_dataset_file
    )

    output_paths = [Path(output_dir)] * len(test_images)

    test_images = np.array(test_images)
    test_dataset = Dataset(
        images=test_images,
        images_masks=test_labels,
        images_names=test_image_names,
        images_output_dirs=output_paths,
    )

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
        dataset=test_dataset,
        is_evaluate=True,
        col_error_range=None,
        save_foldername=output_dir.absolute(),
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
        boundary_errors=True,
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
