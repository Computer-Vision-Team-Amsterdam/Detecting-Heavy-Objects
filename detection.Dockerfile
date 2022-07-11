# docker build . -f detection.Dockerfile -t epureanudiana/diana-container:bullseye0.1 --platform linux/arm64
# docker run -it epureanudiana/diana-container:bullseye0.1
# change entrypoint to prevent container from exiting, when debugging:
# docker run -it --entrypoint=/bin/bash epureanudiana/diana-container:bullseye0.1
FROM python:3.7.13-bullseye

# Install project dependencies
RUN apt-get update && \
    apt-get -y install \
        ffmpeg \
        libsm6 \
        libxext6


ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 opencv-python
RUN git clone https://github.com/facebookresearch/detectron2.git
RUN python -m pip install -e detectron2


WORKDIR /app
COPY configs/* /app/configs/
COPY inference.py evaluation.py utils.py /app/
# copy weights
COPY model_final.pth /app/
COPY data_sample/ /app/data_sample/

RUN useradd appuser
RUN chown -R appuser /app
RUN chmod 755 /app
USER appuser


ENTRYPOINT ["python", "-u", "inference.py"]

