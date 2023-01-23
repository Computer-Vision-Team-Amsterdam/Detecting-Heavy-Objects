# Use the official Ubuntu 20.04 image as the base image
FROM ubuntu:22.04

# Update the package manager and install the necessary dependencies
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    software-properties-common

# Add the Rclone repository and install Rclone
RUN curl https://rclone.org/install.sh | bash

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