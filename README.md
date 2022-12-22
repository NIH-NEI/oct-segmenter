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

`oct-segmenter generate test -i <path/to/test/dir> -o <directory/to/place/the/test_hdf5_file>`

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

### Evaluation

To evaluate the model with a test dataset use the `oct-segmenter evaluate`
subcommand. It requires a test dataset (HDF5 file) as an input which can be
generated using the `generate test` subcommand.
[See above](#generating-test-dataset-hdf5-file)
The user must configure the evaluation parameters by providing a JSON formatted
configuration file like the following:

```
{
  "graph_search": true
}
```

In addition to generating plots with labels, `evaluate` will create a
`results.csv` and `results.hdf5` that contain the following model's
performance statistics:
- `Mean absolute errors`
- `Mean errors`
- `Median absolute errors`
- `SD abs errors`
- `SD errors`
- `Mean dices`
- `SD dices`


#### Example

```
oct-segmenter evaluate \
    -i test_dataset.hdf5 \
    -c config.json \
    -o evaluation-results/
```

### Evaluation Configurable Parameters
The following is a list of the parameters that can be configured for
evaluation:
- `grap_search`: Perform a graph search based on the model predictions
to generate the delineations of the layers.

## Advanced Usage
### Training

The `oct-segmenter` provides the functionality to train a model from
scratch. The user must input a training dataset generated by the
`oct-segmenter generate train` command
([See above](#generating-training-dataset-hdf5-file)).
The user must configure the training parameters by providing a JSON formatted
configuration file like the following:

```
{
  "model_architecture": "unet",
  "model_hyperparameters": {
    "conv_layers": 3
  },
  "epochs": 1000,
  "batch_size": 3,
  "augment": true,
  "experiment": "my-experiment",
  "tracking_uri": "mlruns",
  "username": "balvisio",
  "password": "changeme" # Not recommended in JSON
}
```

#### Example
The following command shows how to start a training run:
```
oct-segmenter train -i <path/to/training/dataset/hdf5/file> \
  -o <path/to/output> -c <path/to/config/json>
```

### MLflow Support
The `oct-segmenter` supports logging runs using the MLflow library. Runs can
be saved locally by default and can also be saved to a remote tracking server
by setting the `tracking_uri` in the config file
([See Training Configurable Parameters](#training-configurable-parameters)).


For instructions on how to setup a remote MLflow tracking server visit the
following
[GitHub gist](https://gist.github.com/balvisio/2cd02a4f403872893d0f3fd8de1f6fba)


To run the MLflow server locally run: `mlflow ui`.

*MacOS Note*: By default, MLflow runs in port 5000. However, in MacOS Monterey
the `ControlCenter` process, which is a native macOS application, uses port
5000. Thus, you can make the MLflow server listen in another port by
running: `mlflow ui --port <port>`.

### Training Configurable Parameters
The following is a list of the parameters that can be configured for training:
- `model_architecture`: string: The model architecture to use for training.
Current supported values are: `"unet"` and `"deeplabv3+"`.
- `model_hyperparameters`: dict: A dictionary to specify different parameters
of the model architecture. The key/value pairs will be dependent on the
chosen `model_architecture` parameter. The allowed values for:
    - `unet`:
        - `start_neurons`: int
        - `pool_layers`: int
        - `conv_layers`: int
        - `enc_kernel`: list
        - `dec_kernel`: list
- `augment`: bool: If `True`, images are augmented by flipping left-right.
- `batch_size`: int
- `class_weight`: "balanced" | list:
    - "balanced": Calculates classes' weights from the class distribution in
    the training and validation datasets. It will use the
    `compute_class_weight` method from the `scikit-learn` library. ([See here](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html))
    - list: List of length equal to the number of classes. Each element is a
    weighting factor for each of the classes.
- `early_stopping`: bool: Stop training when a monitored metric has stopped
  improving. ([See EarlyStopping](https://keras.io/api/callbacks/early_stopping/))
- `epochs`: int
- `early_stopping`: bool: If ``True`` training will stop when the validation
  dice coefficient stops improving after 5 epochs. Overrides `epochs`
  parameter.
- `experiment`: string: Name of the experiment under which the run will be
  logged in MLflow.
- `loss`: string: Loss function to use during training. Currently supported
functions are: `dice_loss`, `focal_loss`.
- `metric`: string: Metric to monitor during training. Currently supported
metrics are: `dice_coef`.
- `tracking_uri`: string: Tracking URI for logging the run. The URI can either
  be a HTTP/HTTPS URI for a MLflow remote server, a database connection string,
  or a local path to log data to a directory. The URI defaults to `mlruns`.
- `username`: string: MLflow server username.
- `password`: string: MLflow server password.
- `patience`: Number of epochs with no improvement after which training will be
  stopped. (Only applicable when `early_stopping = True`)

## Post-processing (Currently not supported, to be implemented)
The script `merge_images.py` merges the original image withe the segmentation plots from the model evaluation/prediction. Usage:

`python3 merge_image.py <path/to/original/image> <path/to/left/segment/plot> <path/to/right/segment/plot> <path/to/output_file>`

For example:

`python3 merge_image.py ../images/testing/2019.10.23/508_OD_R_1_0_0000097_RegAvg/001.tiff ../../ML-Image-Segmentation/results/2021-09-21_21_26_25_U-net_mice_oct/no\ aug_testing_dataset.hdf5/image_6/seg_plot.png ../../ML-Image-Segmentation/results/2021-09-21_21_26_25_U-net_mice_oct/no\ aug_testing_dataset.hdf5/image_7/seg_plot.png mice4.png`


# Other Information

## No package installation usage

1. Create a conda or virtual environment and activate it:
```
conda create -n oct-env python=3.9
conda activate oct-env
```

2. Install packages using `pip`:
```
pip install -r requirements.txt
```

3. To run `oct-segmenter` from repo without installing packages:

```
python3 run.py <subcommand> [options]
```

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

## Additional Scripts
The repository contains a collection of scripts that were created for a
variety of reasons. The following is a list of them:
- `preprocessing-scripts/custom/trim_upper_and_lower_layers.py`: This script
  takes as input a directory containing TIFF files and their corresponding
  CSVs. It trims the upper and lower layers of the images so that the model
  can focus on all layers. It looks at the whole dataset first to find the
  image with the shortest top layer and uses that as the constraint to trim all
  the dataset. The same procedure is used for the bottom layer.
- `preprocessing-scripts/custom/check_retina_layers_order.py`: This script
  takes as input a directory containing the mask files (in CSV format) and
  checks that the all the expected retina layers are present and in order.
- `preprocessing-scripts/custom/map_image_name_to_subject.py`: This script
  creates a text file that maps an image name provided by VFC or WSU to a
  subject. The generated map has the format: `<image_name> <subject>`.
- `preprocessing-scripts/custom/split_images_into_train_val_test.py`: This
  script takes as input a directory containing TIFF files and their
  corresponding CSVs and a TSV file that maps image name to subject (see
  `preprocessing-scripts/custom/map_image_name_to_subject.py`). It splits the
  images into the training, test and validation datasets making sure that no
  subject appears in more than one partition.
- `preprocessing-scripts/custom/calculate_class_imbalance.py`: This script
  takes as input a directory containing the mask files (in CSV format) and
  calculates the class fractions of each image. It then prints the average
  class fractions across all the images.

# OCT Images Quality Improvement

As part of the development of the `oct-segmenter` and the analysis of OCT
images, the algorithms proposed in the paper "Girard MJ, Strouthidis NG,
Ethier CR, Mari JM. Shadow removal and contrast enhancement in optical
coherence tomography images of the human optic nerve head. Invest Ophthalmol
Vis Sci. 2011;52(10):7738-7748. Published 2011 Sep 29.
doi:10.1167/iovs.10-6925" were implemented in MATLAB. These are the descrete
forms of the algorithm described in the appendix of the paper:
- `preprocessing-scripts/oct-image-contrast-enhancement/comp_exp.m`: Applies
  compensation and then exponentation.
- `preprocessing-scripts/oct-image-contrast-enhancementexp_comp.m`: Applies
  exponentiation and then compensation.
- `preprocessing-scripts/oct-image-contrast-enhancement/comp_only.m`: Applies
  compensation only.

These scripts take a path to a TIFF file as input and save the converted
TIFF to an output file name. To convert all the 'tiff' images in a directory
see the script
`preprocessing-scripts/oct-image-contrast-enhancement/enhance_images_contrast.m`
