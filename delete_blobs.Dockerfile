# az acr build -t delete_blobs:latest -f delete_blobs.Dockerfile -r cvtweuacrogidgmnhwma3zq .
FROM python:3.7

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install \
    azure-identity==1.10.0 \
    azure-keyvault-secrets==4.5.1 \
    azure-storage-blob==12.13.1

WORKDIR /app
COPY delete_blobs.py /app
COPY azure_storage_utils.py /app


RUN useradd appuser
RUN chown -R appuser /app
RUN chmod 755 /app
USER appuser
