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
and `unet-x-py2.py3-none-any.whl`

or

2. Run the `build.sh` scripts located in:
  - `mouse-image-segmentation` directory: this will generate the `oct-segmenter` wheel file under
  the `dist` directory.
  - `unet` directory: this will generate the `unet` wheel file under the `unet/dist` directory.


### Windows

The following steps describe all the steps and dependencies to run `oct-segmenter`:

1. Install Anaconda:
  - Download Anaconda from https://www.anaconda.com/products/individual. At the time of this writing
  the current anaconda lives in https://repo.anaconda.com/archive/Anaconda3-2021.05-Windows-x86_64.exe

2. Open the Conda Powershell and create a new Conda environment:

`conda create --name <env_name> python=3.8`

For example:

`conda create --name oct-segmenter-env python=3.8`

3. Install the `oct-segmenter` python package:

There are two ways to install `oct-segmenter`:

  - Installing each package separately:

    3.a.i. `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <path/to/unet/wheel/file>`

    For example:

    `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org ~/unet-1.0-py2.py3-none-any.whl`

    3.a.ii. `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <path/to/oct-segmenter/wheel/file>`

    For example:

    `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org ~/oct_segmenter-1.0-py2.py3-none-any.whl`

or

  - Install both packages simultaneously:

    `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <path/to/oct-segmenter/wheel/file> --no-index --find-links file://<path/to/unet/wheel/package>`

  For example:
  
    `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org ~/oct_segmenter-1.0-py2.py3-none-any.whl --no-index --find-links file://~/unet-1.0-py2.py3-none-any.whl` 


## Usage

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

#### Expected output

The 

### Generating Training Dataset (Currently not supported, to be implemented)

`generate_training_dataset.py` creates a training set by specifying the directories where the training and validation images live. The script will walk recursively inside the tree and include all images that it finds along the way.  The script takes the following parameters:
- `<path/to/dir/with/training/images>`: This is the root directory to pick images (`.tiff`, `.bmp` files) that will be used for training.
- `<path/to/dir/with/validation/images>`: This is the root directory to pick images (`.tiff`, `.bmp` files) that will be used for validation.
- `<path/to/output_file_name>`: Name of the generated HDF5 file.

Usage:

`python3 generate_training_dataset.py <path/to/training/hdf5> <path/to/validation/hdf5> <path/to/output_file_name>`

For example:

`python3 generate_training_dataset.py ../images/training/ ../images/validation/
../train_dataset`TODO


### Generating Testing Dataset (Currently not supported, to be implemented)

The script takes a dataset generated by `generate_dataset.py` script and converts it to a `testing_dataset.py`. Usage:

`python3 generate_test_dataset.py <path/to/test/dir> <path/to/output_file_name>`

For example:

`python3 generate_test_dataset.py ../images/testing/ example-output/test_dataset.hdf5`

## Post-processing (Currently not supported, to be implemented)

The script `merge_images.py` merges the original image withe the segmentation plots from the model evaluation/prediction. Usage:

`python3 merge_image.py <path/to/original/image> <path/to/left/segment/plot> <path/to/right/segment/plot> <path/to/output_file>`

For example:

`python3 merge_image.py ../images/testing/2019.10.23/508_OD_R_1_0_0000097_RegAvg/001.tiff ../../ML-Image-Segmentation/results/2021-09-21_21_26_25_U-net_mice_oct/no\ aug_testing_dataset.hdf5/image_6/seg_plot.png ../../ML-Image-Segmentation/results/2021-09-21_21_26_25_U-net_mice_oct/no\ aug_testing_dataset.hdf5/image_7/seg_plot.png mice4.png`


# Other Information

## Repo Contents

## Preprocess

The script `preprocess.py` labels and creates segmentation maps from a given image. It outputs 4 files:
- <image_name>_{left,right}.json: Two sections of the original image are cropped and labeled. These files are `labelme`
compatible JSON files.
- <image_name>_{left,right}_label.png: These files are the segmentation maps corresponding to the images above.

Usage:

`python preprocess.py </path/to/image> </path/to/output/dir>`

Note: The script assumes that given the path to an input image, a corresponding CSV files with the labels for each layer will
be present in the ◊same directory.

## HDF5 file

The script `generate_dataset.py` generates a single hdf5 file. Usage:

`python generate_dataset.py </path/to/input/dir> </path/to/output/hdf5/file>`

The hdf5 file will contain 3 datasets:
- x: Original images
- y: Segmentation maps of `x`
- image_names: The path to the image files corresponding to `x`

For example:

`python3 generate_dataset.py ../images/training/ example-output/training_dataset.hdf5`


## Building Wheel

To build the `oct-segmenter` wheel package, from the root directory do

```
python setup.py bdist_wheel --universal
```
