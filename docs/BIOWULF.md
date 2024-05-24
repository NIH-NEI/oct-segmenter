# Biowulf

It is possible to use the `oct-segmenter` in Biowulf by downloading the
Singularity image published in the GitHub repo. The following instructions show
how to use the `oct-segmenter` in Biowulf. The steps assume some familiarity
with the Linux console.

> [!WARNING]
> As of 5/23/24, new Singularity images were disabled due to issues with the
> build process. These instructions are retained in case it is re-enabled in
> the future.


## Environment Setup

### Setting up Singularity Directories

Singularity uses the directory `$HOME/.singularity/cache` to store downloaded
or generated SIF container images by default. Since this directory might grow
in size and the space in `$HOME` is limited, it is first recommended to change
it to the `/data` partition. The steps are:

1. Run the following command to set the `SINGULARITY_CACHEDIR`:

   ```
   echo "export SINGULARITY_CACHEDIR=/data/$USER/.singularity/cache" >> .bash_profile
   ```

2. For the change to take effect either logout and login again or execute:

   ```
   source .bash_profile
   ```

   Also it is a good idea to create a dedicated directory to store the singularity
   images in the `/data` directory:

   ```
   mkdir /data/$USER/singularity-images
   ```

### Creating a GitHub Personal Access Token (PAT)

A GitHub PAT is required to download the singularity image into Biowulf. The
steps required to create it are:

Go to your GitHub account and click on the your profile icon on top right of
the screen. Once the dropdown menu appears go to "Settings":

![readme-images/github_pat_1.png](readme-images/github_pat_1.png)


### Starting a Interactive Session

Start an interactive session in a compute node in Biowulf. An example command
that requests a node with one NVIDIA K80 GPU:

```
sinteractive --tunnel --gres=gpu:k80:1 --cpus-per-task=8 --mem=32g --time=24:00:00
```

### Load Singularity and Download the Singularity Image

```
module load singularity
singularity registry login --username <github_username> oras://ghcr.io
```

You will be prompted for the GitHub PAT created in the previous section.

Then the oct-segmenter singularity image can be downloaded as:

```
singularity pull --dir /data/$USER/singularity-images oras://ghcr.io/nih-nei/oct-segmenter-singularity:master
```

Detailed information on interactive jobs can be in the [Biowulf User Guide](https://hpc.nih.gov/docs/userguide.html).



