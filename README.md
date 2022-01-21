# oct-segmenter

The `oct-segmenter` is a command line interface (CLI) tool that allows researchers to automatically
label a mouse retina scans. Given an input image, the tool outputs a CSV files with the coordinates
of each of the layers. Internally, the `oct-segmenter` uses a trained U-net ML model based on the
paper "Automatic choroidal segmentation in OCT images using supervised deep learning methods"

Link: https://www.nature.com/articles/s41598-019-49816-4


## Installation

### Package Files

To use `oct-segmenter` you should have:
1. recieved two wheel files from Bioteam named: `oct_segmenter-x-py2.py3-none-any.whl`
and `oct_unet-x-py2.py3-none-any.whl`

or

2. Run the `build.sh` scripts located in:
  - `mouse-image-segmentation` directory: this will generate the `oct-segmenter` wheel file under
  the `dist` directory.
  - `unet-mod` directory: this will generate the `oct_unet` wheel file under the `unet-mod/dist` directory.


### Windows

The following steps describe all the steps and dependencies to run `oct-segmenter`:

1. Install Anaconda:
  - Download Anaconda from https://www.anaconda.com/products/individual. At the time of this writing
  the current anaconda lives in https://repo.anaconda.com/archive/Anaconda3-2021.05-Windows-x86_64.exe

2. Open the Conda Powershell and create a new Conda environment:

`conda create --name <env_name> python=3.9`

For example:

`conda create --name oct-segmenter-env python=3.9`

3. Install the `oct-segmenter` python package:

There are two ways to install `oct-segmenter`:

  - Installing each package separately:

    3.a.i. `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <path/to/oct_unet/wheel/file>`

    For example:

    `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org ~/oct_unet-0.3.0-py2.py3-none-any.whl`

    3.a.ii. `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <path/to/oct_segmenter/wheel/file>`

    For example:

    `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org ~/oct_segmenter-1.0-py2.py3-none-any.whl`

or

  - Install both packages simultaneously:

    `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <path/to/oct-segmenter/wheel/file> --no-index --find-links file://<path/to/unet/wheel/package>`

  For example:
  
    `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org ~/oct_segmenter-1.0-py2.py3-none-any.whl --no-index --find-links file://~/unet-1.0-py2.py3-none-any.whl` 


## Usage

### Generating Datasets

### Partitioning Input Images into datasets

`oct-segmenter partition` can be used to partition a directory containing the images and CSVs into
the training, validation and test datasets. It creates a random permutation and partitions the
images according to the fractions passed in the `--training`, `--validation` and `--test` flags.
Defaults are: training: `0.56`, validation: `0.14`, test: `0.3`.
The `oct-segementer` requires that the names of image files and their corresponding CSVs to be
indentical.

#### Example

`oct-segmenter partition -i <path/to/input/images> -o <path/to/output/partition> --training 0.3
--validation 0.5 --test 0.2`


### Generating Training Dataset HDF5 File

Given one directory containing images to be used for training and a second one with images
containing images used for validation, `oct-segmenter generate training` creates a HDF5 file named
`training_dataset.hdf5` that contains the images used for training and validation in a format that
can be directly be used to train an `oct-unet` model.
The `oct-segementer` requires that the names of image files and their corresponding CSVs to be
indentical.

#### Example

`oct-segmenter generate training --training-input-dir <path/to/training/dir> --validation-input-dir
<path/to/validation/dir> -o <directory/to/place/the/training_hdf5_file>`


### Generating Test Dataset HDF5 File

Given one directory containing images to be used for testing `oct-segmenter generate test` creates
a HDF5 file named `test_dataset.hdf5` that contains the images used for testing in a format that
can be directly be used for evaluation using an already trained `oct-unet` model.
The `oct-segementer` requires that the names of image files and their corresponding CSVs to be
indentical.

#### Example

`oct-segmenter generate test -t <path/to/test/dir> -o <directory/to/place/the/test_hdf5_file>`

### Listing and selecting available models
`oct-segmenter` is packaged with trained models. Currently the included models are:
- `visual-function-core`: Model trained with images provided by the NIH-NEI Visual Function Core
group.
- `wayne-state-university`: Model trained with images provided by Wayne State University.

To list the available models:
```
oct-segmenter list
```

The command will display the list of available models and the default model to use in the
`predict` and `evaluate` subcommands.
The command will prompt the user to type a number to select a new default model. If no change is desired, press `Enter`.

### Prediction

To generate labels for new images there are two alternatives. Give the following flags:
- `-i`: Give path to the TIFF file to be labeled.
- `-d`: Give path to a directory. The `oct-segmenter` tool will look for all `.tiff` files and
  generate labels for each of them.

#### Examples

```
oct-segmenter predict -i myimage.tiff -o myoutput
oct-segmenter predict -d testing_images
```

### Advanced Usage
### Training

`oct-segmenter` provides the functionality to train an oct-model from scratch. The user must input
a training dataset generated by the `oct-segementer generate train` command. [See above](#generating-training-dataset-hdf5-file)
User can configure training parameters by providing a JSON formatted configuration file like the following:

```
{
  "epochs": 1000,
  "batch_size": 3
}
```

#### Example

```
oct-segmenter train -i <path/to/training/dataset/hdf5/file> -o <path/to/output> -c training-config.json
```

### Training Configurable Parameters
- batch_size
- epochs

## Post-processing (Currently not supported, to be implemented)

The script `merge_images.py` merges the original image withe the segmentation plots from the model evaluation/prediction. Usage:

`python3 merge_image.py <path/to/original/image> <path/to/left/segment/plot> <path/to/right/segment/plot> <path/to/output_file>`

For example:

`python3 merge_image.py ../images/testing/2019.10.23/508_OD_R_1_0_0000097_RegAvg/001.tiff ../../ML-Image-Segmentation/results/2021-09-21_21_26_25_U-net_mice_oct/no\ aug_testing_dataset.hdf5/image_6/seg_plot.png ../../ML-Image-Segmentation/results/2021-09-21_21_26_25_U-net_mice_oct/no\ aug_testing_dataset.hdf5/image_7/seg_plot.png mice4.png`


# Other Information

## No package installation usage
To run `oct-segmenter` from repo without installing packages

`python3 run.py`

#### Example

```
python3 run.py predict -d images/
```

## Preprocess

The script `preprocess.py` labels and creates segmentation maps from a given image. It outputs 4 files:
- <image_name>_{left,right}.json: Two sections of the original image are cropped and labeled. These files are `labelme`
compatible JSON files.
- <image_name>_{left,right}_label.png: These files are the segmentation maps corresponding to the images above.

Usage:

`python preprocess.py </path/to/image> </path/to/output/dir>`

Note: The script assumes that given the path to an input image, a corresponding CSV files with the labels for each layer will
be present in the ◊same directory.


## Building Python Wheel Package

To build the `oct-segmenter` wheel package, from the root directory do

```
./build.sh
```
