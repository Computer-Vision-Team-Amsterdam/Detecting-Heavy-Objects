# Build with following command to forward SSH keys from host machine to access private GitHub repo's:
# DOCKER_BUILDKIT=1 docker build --ssh default -f postprocessing.dockerfile -t as01weuacrovja4xx7veccq.azurecr.io/container-detection-postprocessing:0.1 .
# This should be addressed when building this image in a pipeline

FROM python:3.7

RUN apt-get update && \
    apt-get -y install \
        ffmpeg \
        gdal-bin \
        libgdal-dev \
        libsm6 \
        libxext6

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python -m pip install --upgrade pip

RUN pip install --no-cache \
        gdal==3.2.2 \
        geojson==2.5.0 \
        geopy==2.2.0 \
        numpy==1.21.6 \
        pycocotools==2.0.4 \
        requests==2.28.0 \
        pandas==1.3.5 \
        psycopg2==2.8.6 \
        scipy==1.7.3 \
        shapely==1.8.2 \
        tqdm==4.64.0 \
        azure-cli==2.39.0 \
        azure-identity==1.10.0 \
        azure-keyvault-secrets==4.5.1 \
        azure-storage-blob==12.13.1 \
        git+https://git@github.com/Computer-Vision-Team-Amsterdam/Geolocalization_of_Street_Objects.git@faba8f6f4a94e545135dd24aea398defe2c69f97 \
        git+https://git@github.com/Computer-Vision-Team-Amsterdam/panorama.git@v0.2.3

WORKDIR /app
COPY visualizations/stats.py visualizations/utils.py /app/visualizations/
COPY postprocessing.py /app
COPY azure_storage_utils.py /app
COPY upload_to_postgres.py /app

RUN useradd appuser
RUN chown -R appuser /app
USER appuser
