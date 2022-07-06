FROM python:3.7.13-bullseye


# Install project dependencies

RUN apt-get update && \
    apt-get -y install \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgeos-dev # used by shapely \


ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 opencv-python
RUN git clone https://github.com/facebookresearch/detectron2.git
RUN python -m pip install -e detectron2



WORKDIR /opt
COPY configs/config_parser.py configs/container_detection.yaml /opt/configs/
COPY inference.py /opt
COPY evaluation.py /opt
COPY utils.py /opt

RUN useradd appuser
USER appuser

ENTRYPOINT ["python", "-u", "inference.py"]

