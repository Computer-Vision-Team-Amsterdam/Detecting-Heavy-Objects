# Build with following command to forward SSH keys from host machine to access private GitHub repo's:
# DOCKER_BUILDKIT=1 docker build --ssh default -f postprocessing.dockerfile -t as01weuacrovja4xx7veccq.azurecr.io/container-detection-postprocessing:0.1 .
# This should be addressed when building this image in a pipeline

FROM python:3.7.13-bullseye

RUN apt-get update && \
    apt-get -y install \
        ffmpeg \
        gdal-bin \
        libgdal-dev \
        libsm6 \
        libxext6

RUN  mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

RUN --mount=type=ssh \
    pip install \
    gdal==3.2.2 \
    geojson==2.5.0 \
    geopy==2.2.0 \
    numpy==1.21.6 \
    pycocotools==2.0.4 \
    requests==2.28.0 \
    shapely==1.8.2 \
    tqdm==4.64.0 \
    git+ssh://git@github.com/Computer-Vision-Team-Amsterdam/Geolocalization_of_Street_Objects.git@d88bea3fd9f26028a744700b225ef11c328bd3de \
    git+ssh://git@github.com/Computer-Vision-Team-Amsterdam/panorama.git@98a92686a9ef92b3748f345b137123ea5915c8b1

WORKDIR /opt
COPY visualizations/stats.py visualizations/utils.py /opt/visualizations/
COPY postprocessing.py /opt
