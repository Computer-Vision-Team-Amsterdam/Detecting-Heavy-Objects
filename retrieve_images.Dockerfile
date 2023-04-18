# az acr build -t retrieve:latest -f retrieve_images.Dockerfile -r cvtweuacrogidgmnhwma3zq .
FROM python:3.7


ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install \
    requests==2.26.0 \
    pandas \
    azure-cli==2.39.0 \
    azure-identity==1.10.0 \
    azure-keyvault-secrets==4.5.1 \
    azure-storage-blob==12.13.1

RUN mkdir -p /opt/retrieved_images
WORKDIR /opt
COPY retrieve_images.py /opt
COPY utils /opt/utils

RUN useradd appuser
# needed in this case to get access look through the folders
RUN chown -R appuser /opt
RUN chmod 755 /opt
USER appuser