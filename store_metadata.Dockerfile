# DOCKER_BUILDKIT=1 docker build . --ssh default -f store_metadata.Dockerfile -t cvtweuacrogidgmnhwma3zq.azurecr.io/store_metadata:latest --platform linux/amd64
FROM python:3.7.13-bullseye

# RUN  mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install \
    azure-identity==1.10.0 \
    azure-keyvault-secrets==4.5.1 \
    azure-storage-blob==12.13.1 \
    psycopg2==2.8.6 \
    git+http://git@github.com/Computer-Vision-Team-Amsterdam/panorama.git@98a92686a9ef92b3748f345b137123ea5915c8b1

WORKDIR /opt
COPY metadata_to_postgresql.py /opt

RUN useradd appuser
USER appuser
