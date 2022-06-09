FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04 AS base-image

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
RUN python -m pip install azureml-sdk
RUN python -m pip install shapely
RUN python -m pip install -U albumentations
RUN python -m pip install wandb




# Create working directory
RUN  mkdir app
WORKDIR /app

# Copy contents
COPY . /app

CMD ["python", "-u", "inference.py", "--name", "detectron_map2", "--version", "2", "--device", "cpu", "--subset", "test"]