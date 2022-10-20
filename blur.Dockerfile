# Partly copied from https://github.com/ultralytics/yolov5/blob/master/utils/docker/Dockerfile-cpu
# az acr build -t blur:latest -f blur.Dockerfile -r cvtweuacrogidgmnhwma3zq .
FROM python:3.7.15-bullseye

# Install linux packages
RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx

# Elegant method, virtualenv automatically works for both RUN and CMD
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install pip packages
COPY yolov5/requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache -r requirements.txt 

# Create working directory
WORKDIR /app

# Copy contents (NOTE: dont copy weights file in docker image in the future)
COPY yolov5  /app
COPY azure_storage_utils.py /app

# Installation as root, non-root user when executing the container

RUN useradd appuser
RUN chown -R appuser /app
USER appuser