# Use an official Alpine Linux as a parent image
FROM ubuntu:20.04

# Install rclone
RUN apk add --no-cache ca-certificates openssl unzip \
    && wget -q https://downloads.rclone.org/rclone-current-linux-amd64.zip \
    && unzip rclone-current-linux-amd64.zip \
    && mv rclone-*-linux-amd64/rclone /usr/bin/ \
    && rm -r rclone-*-linux-amd64*

# Packages required to run the Azure CLI installation
RUN	apt-get update && apt-get -y install curl

# Azure installation command
RUN	curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# Copy the bash script to the container
WORKDIR /opt
COPY retrieve_images.sh /opt

# Make the script executable
RUN chmod +x /opt/retrieve_images.sh

#RUN useradd appuser
# needed in this case to get access look through the folders
#RUN chown -R appuser /opt
#RUN chmod 755 /opt
#USER appuser