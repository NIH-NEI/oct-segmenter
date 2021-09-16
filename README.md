# Image Preprocessing

This repository contains the scripts to preprocess the images shared by NIH-NEI.

The processed images can be used as input to the model in NIH-NEI/ML-Image-Segmentation

# Repo Contents

## Preprocess

The script `preprocess.py` labels and creates segmentation maps from a given image. It outputs 4 files:
- <image_name>_{left,right}.json: Two sections of the original image are cropped and labeled. These files are `labelme`
compatible JSON files.
- <image_name>_{left,right}_label.png: These files are the segmentation maps corresponding to the images above.

Usage:

`python preprocess.py </path/to/image> </path/to/output/dir>`

Note: The script assumes that given the path to an input image, a corresponding CSV files with the labels for each layer will
be present in the same directory.

## HDF5 file

The script `generate_dataset.py` generates a single hdf5 file. Usage:

`python generate_dataset.py </path/to/input/dir> <HDF5 file name> </path/to/output/dir>`

The hdf5 file will contain 3 datasets:
- x: Original images
- y: Segementation maps of `x`
- image_names: The name of the image files corresponding to `x`

For example:

`python3 generate_dataset.py ../images/training/ training_dataset  example-output`

## Merge Datasets

This script merges two datasets together. Usage:

`python merge_datasets.py <path/to/first/hdf5> <path/to/second/hdf5> <dataset name 1> <dataset name 2> <dataset name 3> <dataset name 4> <output_file_name> <output file dir>`

For example:

`python3 merge_datasets.py ../images/training/training_dataset.hdf5 ../images/validation/validation_dataset.hdf5 train_images train_labels val_images val_labels train_dataset ../`

# Environment Dependencies

The file `requirements.txt` contains the list of dependencies. Can be used to create the environment with:

`conda create --name <env_name> --file requirements.txt`