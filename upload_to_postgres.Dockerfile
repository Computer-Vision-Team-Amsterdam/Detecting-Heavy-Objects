# az acr build -t upload_to_postgres:latest -f upload_to_postgres.Dockerfile -r cvtweuacrogidgmnhwma3zq .
FROM python:3.7

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install \
    azure-identity==1.10.0 \
    azure-keyvault-secrets==4.5.1 \
    azure-storage-blob==12.13.1 \
    psycopg2==2.8.6 \
    git+https://git@github.com/Computer-Vision-Team-Amsterdam/panorama.git@v0.2.3

WORKDIR /opt
COPY upload_to_postgres.py /opt
COPY utils /opt/utils


RUN useradd appuser
RUN chown -R appuser /opt
RUN chmod 755 /opt
USER appuser
