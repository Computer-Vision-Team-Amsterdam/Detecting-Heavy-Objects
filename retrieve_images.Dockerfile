# docker build . -f retrieve_images.Dockerfile -t epureanudiana/retrieve-images --platform linux/arm64

# change entrypoint to prevent container from exiting, when debugging:
# docker run --env KEY_VAULT_NAME=${KEY_VAULT_NAME} -it --entrypoint=/bin/bash epureanudiana/retrieve-images

FROM python:3.7.13-bullseye

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install \
    requests==2.26.0 \
    azure-cli==2.39.0 \
    azure-identity==1.10.0 \
    azure-keyvault-secrets==4.5.1

WORKDIR /opt
COPY retrieve_images.py /opt
COPY retrieved_images /opt/retrieved_images/

RUN useradd appuser
# needed in this case to get access look through the folders
RUN chown -R appuser /opt
RUN chmod 755 /opt
USER appuser