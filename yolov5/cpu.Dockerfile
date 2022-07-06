# Partly copied from https://github.com/ultralytics/yolov5/blob/master/utils/docker/Dockerfile-cpu
FROM python:3.7.13-bullseye

# Install linux packages
RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx

# Install pip packages
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip uninstall -y torch torchvision torchtext Pillow
RUN pip install --no-cache -r requirements.txt 

# Create working directory
RUN  mkdir app
WORKDIR /app

# Copy contents
COPY . /app