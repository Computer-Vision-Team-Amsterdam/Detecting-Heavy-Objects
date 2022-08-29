# docker build . -f connection.Dockerfile -t epureanudiana/store-postprocessing-results --platform linux/arm64

# change entrypoint to prevent container from exiting, when debugging:
# docker run -it --entrypoint=/bin/bash epureanudiana/store-postprocessing-results
# docker run --env USERNAME=${USERNAME} --env PASSWORD=${PASSWORD} --env HOST=${HOST}
# --env AZURE_STORAGE_CONNECTION_STRING=${AZURE_STORAGE_CONNECTION_STRING}  -it --entrypoint=/bin/bash epureanudiana/store-postprocessing-results

FROM python:3.7.13-bullseye

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install \
    psycopg2==2.8.6 \
    pandas==1.3.4 \
    azure-storage-blob==12.13.1

WORKDIR /opt
COPY connect.py /opt
COPY prioritized_objects.csv /opt

RUN useradd appuser
USER appuser