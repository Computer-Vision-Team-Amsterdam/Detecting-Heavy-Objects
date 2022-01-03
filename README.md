# Detecting-Heavy-Objects

The goal of the project is to detect heavy objects in the City of Amsterdam using the Panorama images. 

## Requirements

- A recent version of Python 3. The project is being developed on Python 3.9, but should be compatible with some older minor versions.
- This project uses [Poetry](https://python-poetry.org/) as its package manager.

## Getting Started
Clone this repository and install the dependencies:

```shell
git clone https://github.com/Computer-Vision-Team-Amsterdam/Detecting-Heavy-Objects.git
poetry install
poetry run python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
## Training the model 

To vizualize training loss curves, accuracies, run
```shell
python -m tensorboard.main --logdir $PWD/output
```