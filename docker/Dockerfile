FROM tensorflow/tensorflow:2.9.0-gpu

LABEL org.opencontainers.image.authors="Bruno Alvisio <bruno@bioteam.net>"
LABEL org.opencontainers.image.source=https://github.com/NIH-NEI/oct-segmenter

# Next 3 lines are needed due to https://github.com/NVIDIA/nvidia-docker/issues/1632
# RUN apt-key del 7fa2af80 && \
#     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
#     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub && \
#     apt-get update

# Install Python 3.10 and Git
# Clean up apt cache
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.10 python3.10-distutils git && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip
RUN python3.10 -m pip install tensorflow[and-cuda]==2.9.0

# Surface-distance package (https://github.com/deepmind/surface-distance)
# is not in PyPI
RUN pip install git+https://github.com/deepmind/surface-distance.git

COPY *.whl /oct-segmenter-dist/
RUN pip install /oct-segmenter-dist/*.whl

# We set $HOME to a writable location to write config files.
ENV HOME /tmp

ENTRYPOINT [ "oct-segmenter" ]
