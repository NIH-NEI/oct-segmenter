import os
import sys

import logging as log
from pathlib import Path
import re


if __name__ == "__main__":
    """
    This script creates a text file that maps an image name provided by VFC \
    or WSU to a subject. The generated map has the format:
    <image_name> <subject>

    This map is then used to partition the dataset into training, validation \
    and test.

    Example:
    python preprocessing-scripts/custom/map_image_name_to_subject.py \
        data/experiment-10/images/
    """
    # Set logging
    log.basicConfig()
    log.getLogger().setLevel(log.INFO)

    input_dir = Path(sys.argv[1])

    patterns = []
    # 2019.02.21%2737_OD_R_1_0_0000007_RegAvg%001.tiff
    patterns.append(
        re.compile("\d{4}.\d{2}.\d{2}%([A-Za-z0-9]+)_.*.(?:tiff|TIFF)")
    )
    # 2129 light_2129_OD_V_2x0_90_0000016 RK.tiff
    patterns.append(re.compile("(\d{4}) .*.(?:tiff|TIFF)"))
    # 042321b_Center 1 @913 RK.tiff
    patterns.append(re.compile("(\d{6}\D{1})(?:_| |-).*.(?:tiff|TIFF)"))
    # E100720 - KO #1130909_center 1 @ 1110 -RW RW.tiff
    patterns.append(re.compile("(\D{1}\d{6})(?:_| |-).*.(?:tiff|TIFF)"))
    # Light_2942_OS_R_1_0_0000022_RegAvg BAB RW.tiff
    # PBS-Dark_2940_OS_R_1_0_0000082_RegAvg CG.tiff
    patterns.append(
        re.compile("(?:PBS-|)(?:Dark|Light)_([A-Za-z0-9]+)_.*.(?:tiff|TIFF)")
    )

    total_images = 0
    matches = 0
    non_matches = 0
    filename_to_subject = {}

    for _, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(".tiff") and not filename.startswith(
                "."
            ):
                total_images += 1
                match_found = False
                for pattern in patterns:
                    match = pattern.match(filename)
                    if match:
                        if len(match.groups()) != 1:
                            print(match.groups())
                        assert len(match.groups()) == 1
                        filename_to_subject[filename] = match.groups()[0]
                        matches += 1
                        match_found = True
                        break
                if not match_found:
                    non_matches += 1
                    log.warning(
                        f"Filename: {filename} didn't match any pattern"
                    )

    log.info(f"Found {total_images} TIFF images")
    log.info(f"Found {matches} matches")
    log.info(f"{non_matches} file could not be matched")

    with open(
        input_dir / Path("filename_to_subject.tsv"), "w"
    ) as output_map_file:
        for filename, subject in filename_to_subject.items():
            output_map_file.write(filename + "\t" + subject + "\n")
