from __future__ import annotations

import h5py
from pathlib import Path
from typing import List, Optional
from typeguard import typechecked

from oct_segmenter.preprocessing import generic_dataset as generator


@typechecked
def generate_test_dataset(
    test_input_dir: Path,
    output_file: Path,
    input_format: str,
    rgb_format: bool,
    layer_names: Optional[List[str]],
) -> h5py.File:
    test_hdf5_file = generator.generate_generic_dataset(
        test_input_dir,
        output_file,
        input_format,
        rgb_format,
        layer_names,
    )

    test_hdf5_file["test_images"] = test_hdf5_file["xhat"]
    test_hdf5_file["test_labels"] = test_hdf5_file["yhat"]
    test_hdf5_file["test_segs"] = test_hdf5_file["segs"]
    test_hdf5_file["test_images_source"] = test_hdf5_file["image_source"]

    del test_hdf5_file["xhat"]
    del test_hdf5_file["yhat"]
    del test_hdf5_file["segs"]
    del test_hdf5_file["image_source"]

    return test_hdf5_file
