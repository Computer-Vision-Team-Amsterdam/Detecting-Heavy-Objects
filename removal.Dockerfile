# Remove panorama images from the container based on list of panorama names

# docker build . -f removal.Dockerfile -t epureanudiana/remove_panoramas --platform linux/arm64
# docker run -it epureanudiana/remove_panoramas
# change entrypoint to prevent container from exiting, when debugging:
# docker run -it --entrypoint=/bin/bash epureanudiana/remove_panoramas

FROM python:3.7.13-bullseye

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
COPY remove_panoramas.py /app
COPY outputs/to_delete/* /app/outputs/to_delete/
COPY outputs/empty_predictions.json /app/outputs/


