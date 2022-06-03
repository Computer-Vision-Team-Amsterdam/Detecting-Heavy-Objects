# Start FROM NVIDIA PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
#FROM nvcr.io/nvidia/pytorch:22.04-py3
#RUN rm -rf /opt/pytorch  # remove 1.2GB dir
FROM python:3.7

# Install linux packages
RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx

# Install pip packages
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip uninstall -y torch torchvision torchtext Pillow
RUN pip install --no-cache -r requirements.txt 

# Pillow>=9.1.0 torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

# Create working directory
RUN  mkdir app
WORKDIR /app

# Copy contents
COPY . /app

CMD ["python", "-u", "blur.py"]