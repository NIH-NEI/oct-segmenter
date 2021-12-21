import os

import numpy as np
from pathlib import Path

from oct_segmenter import __version__
from oct_segmenter.preprocessing import preprocess

from unet.model import augmentation as aug
from unet.model import evaluation
from unet.model.evaluation_parameters import EvaluationParameters, PredictionDataset
from unet.model.save_parameters import SaveParameters


MODEL = Path(os.path.dirname(os.path.abspath(__file__)) + "/../data/model/" + __version__ + "/model.hdf5")
def predict(args):
    input_paths = []
    if args.input:
        path = Path(args.input)
        if not path.is_file():
            print("oct-segmenter: Input file not found. Exiting...")
            exit(1)
        input_paths.append(path)
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

    pred_images = []
    pred_images_names = []
    output_paths = []
    for input_path in input_paths:
        if args.output:
            output = Path(args.output)
        else:
            output = input_path.parent

        if args.c:
            img = preprocess.generate_input_image(input_path)
            pred_images.append(img)
            pred_images_names.append(Path(input_path.stem + "_labeled" + input_path.suffix))
            output_paths.append(output)
        else:
            img_left, img_right = preprocess.generate_side_region_input_image(input_path)
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

    save_params = SaveParameters(
        pngimages=True,
        raw_image=True,
        raw_labels=True,
        temp_extra=True,
        boundary_maps=True,
        area_maps=True,
        comb_area_maps=True,
        seg_plot=True
    )

    eval_params = EvaluationParameters(
        model_file_path=MODEL,
        prediction_dataset=pred_dataset,
        is_evaluate=False,
        col_error_range=None,
        save_foldername=output_paths[0].absolute(), # TODO: FIX
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
