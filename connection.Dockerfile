# docker build . -f connection.Dockerfile -t epureanudiana/postgres-connection --platform linux/arm64

# change entrypoint to prevent container from exiting, when debugging:
# docker run -it --entrypoint=/bin/bash epureanudiana/postgres-connection
FROM python:3.7.13-bullseye

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install \
    psycopg2==2.8.6 \
    pandas==1.3.4

WORKDIR /opt
COPY connect.py /opt
COPY prioritized_objects.csv /opt

RUN useradd appuser
USER appuser