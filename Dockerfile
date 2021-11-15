FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04 AS base-image
# FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04 AS base-image
# FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04  ## what they use in detectron2 dockerfile

# Upgrade and install system libraries
RUN apt-get -y update \
    && ACCEPT_EULA=Y apt-get upgrade -qq \
    && apt-get -y install \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install project dependencies
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install torch==1.8.0 torchaudio==0.8.0 torchvision==0.9.0 opencv-python
RUN git clone https://github.com/facebookresearch/detectron2.git
RUN python -m pip install -e detectron2


# Continue from clean Python base image
FROM base-image AS final

# COPY --from=base-image $POETRY_HOME $POETRY_HOME
# COPY --from=base-image $PYSETUP_PATH $PYSETUP_PATH
# COPY . .
