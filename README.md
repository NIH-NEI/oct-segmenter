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

To evaluate the model with a test dataset use the `oct-segmenter evaluate` subcommand. It requires
a test dataset (HDF5 file) as an input which can be generated using the `generate test` subcommand.
[See above](#generating-test-dataset-hdf5-file)
In addition to generating plots with labels, `evaluate` will create a `results.csv` and
`results.hdf5` that contain the following model's performance statistics:
- `Mean absolute errors`
- `Mean errors`
- `Median absolute errors`
- `SD abs errors`
- `SD errors`
- `Mean dices`
- `SD dices`


#### Example

```
oct-segmenter evaluate -i current/wayne_state_test_dataset.hdf5  -o current/test-results/
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
- `batch_size`
- `epochs`

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
be present in the same directory.


## Building Python Wheel Package

To build the `oct-segmenter` wheel package, from the root directory do

```
./build.sh
```

## Running in an HPC environment

These instructions are intended for use on Biowulf, but the process should be
similar for any HPC environment using Slurm and Singularity. 

### Building oct-segmenter with Singularity

To begin, copy the Singularity definition file into a new directory. Note:
don't build the Singularity image within your repo.

```
cd ~
mkdir oct-segmenter-sif
cp path-to-repo/singularity/oct-segmenter.def oct-segmenter-sif/
```

Start an interactive session. Then change into the `oct-segmenter-sif`
directory and build a singularity image from the `oct-segmenter.def` file like
this:

```
sinteractive --constraint=gpuk80 --gres=gpu:k80:1 --mem=32GB
cd oct-segmenter-sif
module load singularity
singularity build --remote --sandbox oct-segmenter/ oct-segmenter.def
```

Now that the sandbox environment is done building, drop into an interactive
shell within the Singularity image, copy the repo into the image, and set
permissions.

```
singularity shell --writable oct-segmenter/
cp path-to-repo/ /opt/
chmod -R 577 /opt/path-to-repo
exit
```

Now you need to convert the sandbox into a SIF file.

```
singularity build oct-segmenter-0.3.0.sif oct-segmenter/
```

### Running the oct-segmenter with Singularity on Biowulf

You only need to build the image once, but you will need to enter an
interactive session, load the Singularity module, and set your Singularity
bindpaths everytime you use the tool. If you're continuing immediately from the
build step, you can skip the first two commands and just run the third.
Otherwise, run all three commands:

```
sinteractive --constraint=gpuk80 --gres=gpu:k80:1 --mem=32GB
module load singularity
. /usr/local/current/singularity/app_conf/sing_binds
```

Simply type the following command to run the CLI tool:

```
singularity run --nv oct-segmenter-0.3.0.sif
```

You can pass additional command and arguments as you normally would with the
CLI tool.  For example, to run the `list` command, do the following:

```
singularity run --nv oct-segmenter-0.3.0.sif list
```

### Submitting a job to Slurm

Jobs on Biowulf are scheduled via Slurm. A sample Slurm script is included in
the `slurm/oct-segmenter.sh`.

You will need to specify the path to your `oct-segmenter-0.3.0.sif` file
(`SIF_PATH`) and input directory. Optionally, you may also want to change the
other directories.

The script is configured to partition, generate training, and train, so if you
are using it for anything else, you will need to change these commands as well:

```
$SIF_PATH partition -i $INPUT_DIR -o $PARTITION_DIR --training 0.3 --validation 0.5 --test 0.2
$SIF_PATH generate training --training-input-dir $TRAINING_INPUT_DIR --validation-input-dir $VALIDATION_INPUT_DIR -w -o $TRAINING_HDF5_DIR
$SIF_PATH train -i $TRAINING_HDF5_DIR/training_dataset.hdf5 -o $TRAINING_OUTPUT_DIR -c training-config.json
```

You may want to change the SBATCH commands too, particularly the time, gpus,
and memory. Set your time to be slightly more than your maximum expected run
time. For instance, if your jobs usually take about an hour but never exceed an
hour and half, then you could set your time like this:

```
#SBATCH --time=01:30:00
```

Once you have made all the necessary changes to the Slurm script, run the
following commands:

```
cd path-to-repo/slurm
sbatch oct-segmenter.sh
```

Type `sjobs` to check the status of the job. You should receive an email when
it completes. Output and error logs will be created in the directory from which
you ran the sbatch command. If your job fails, be sure to check these to debug
the issue.
