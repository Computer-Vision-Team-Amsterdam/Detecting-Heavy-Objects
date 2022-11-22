# az acr build -t detection:latest -f detection.Dockerfile -r cvtweuacrogidgmnhwma3zq .

# pull the model directly from aml env
# download the model temporarily from AS during pipeline + copy it when building this dockerfile
# smth like COPY A/model_final.pth .
FROM python:3.7.15-bullseye

# Install project dependencies
RUN apt-get update && \
    apt-get -y install \
        ffmpeg \
        libsm6 \
        libxext6

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install \
    torch==1.10.1 \
    torchvision==0.11.2 \
    torchaudio==0.10.1  \
    opencv-python \
    seaborn \
    azure-identity==1.10.0 \
    azure-keyvault-secrets==4.5.1 \
    azure-storage-blob==12.13.1
RUN git clone https://github.com/facebookresearch/detectron2.git --branch v0.6
RUN python -m pip install -e detectron2


WORKDIR /app
COPY configs/* /app/configs/
COPY inference.py evaluation.py utils.py /app/
# copy weights
# COPY model_final.pth /app/

RUN useradd appuser
RUN chown -R appuser /app
RUN chmod 755 /app
USER appuser

