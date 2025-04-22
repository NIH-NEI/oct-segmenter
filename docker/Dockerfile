FROM tensorflow/tensorflow:2.9.0-gpu

LABEL org.opencontainers.image.authors="Bruno Alvisio <bruno@bioteam.net>"
LABEL org.opencontainers.image.source=https://github.com/NIH-NEI/oct-segmenter

# Next 3 lines are needed due to https://github.com/NVIDIA/nvidia-docker/issues/1632
# RUN apt-key del 7fa2af80 && \
#     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
#     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub && \
#     apt-get update

RUN apt-get update
RUN apt-get install -y git

RUN python -m pip install --upgrade pip

# Surface-distance package (https://github.com/deepmind/surface-distance)
# is not in PyPI
RUN pip install git+https://github.com/deepmind/surface-distance.git

COPY *.whl /oct-segmenter-dist/
RUN pip install /oct-segmenter-dist/*.whl

# We set $HOME to a writable location to write config files.
ENV HOME /tmp

ENTRYPOINT [ "oct-segmenter" ]
