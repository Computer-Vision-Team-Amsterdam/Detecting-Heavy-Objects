# Detecting-Heavy-Objects

The goal of the project is to detect heavy objects in the City of Amsterdam using the Panorama images. 

## Requirements

- A recent version of Python 3. The project is being developed on Python 3.9, but should be compatible with some older minor versions.
- This project uses [conda](/link_to_conda) as its package manager.

## Getting Started
1. Clone this repository:

```shell
git clone https://github.com/Computer-Vision-Team-Amsterdam/Detecting-Heavy-Objects.git
```
2. For the virtual environment, the project uses the [`det2`](./det2.yml) environment.

To create the environment from the .yml file, run 
```shell
conda env create -f det2.yml
```


To activate the environment, run 
```shell
conda activate det2
```

To deactivate the environment, run 
```shell
conda deactivate
```

## Training the model 

For this project we use an instance segmentation framework named [Detectron2](/link_to_detectron).
From the provided [models zoo](/link_to_model_zoo), we start from a pre-trained model which uses the Base-RCNN-FPN 
configuration. The model performs both instance segmentation and object detection on a given image.  


To train the model on your local machine, run 
```shell
python training.py --name &{MODEL_NAME} --device cpu
```

```${MODEL_NAME}``` is the name you give to your model; can be anything.

The output files are stored according to the [configuration file](./configs/container_detection.yaml).
By default, output files are stored in the ```outputs``` folder.


To train the model on GPU, run 
```shell
python run_train_on_azure.py --name ${MODEL_NAME} --subset train

```
Latest model is registered on Azure.
The output files are stored on Azure in the ```outputs``` folder and contain the following:

- *model_final.pth* -- trained pytorch model
- *metrics.json* -- metrics used for plotting by the tensorboard
- *events.out.tfevents* -- output summary used for plotting by tensorboard
- *last_checkpoint* -- stores reference to name of the last trained model.

The output files are also downloaded locally in ```outputs/TRAIN_${MODEL_NAME}_${MODEL_VERSION}``` folder.

Once you have a trained model, you can visualize the training loss curves, accuracies and other metrics by running:
```shell
python -m tensorboard.main --logdir $PWD/outputs/TRAIN_${MODEL_NAME}_${MODEL_VERSION}
```

## Model Inference

To perform inference on your local machine, run
```shell
python inference.py --name ${MODEL_NAME} --version ${MODEL_VERSION} --device cpu

```
Currently, you need to make sure you have the MODEL_NAME with MODEL_VERSION locally as well as the data folder.


To perform inference on the validation set on GPU, run
```shell
python run_inference_on_azure.py --name ${MODEL_NAME} --version ${VERSION} --subset val 
```

To run inference on a test dataset with no corresponding annotation file, use ```--subset test``` instead.

The output files are as follows:
- *container_val_coco_format.json* -- automatically generated COCO file for Detectron2. It contains metadata of the 
dataset used at inference time 
- *coco_instances_results.json* -- predictions of all instances
- *instances_predictions.pth* -- ? 
- *eval_metrics.txt* -- stores evaluation metrics described below

The trained model is downloaded locally at ```azureml-models/${MODEL_NAME}/${MODEL_VERSION}/model_final.pth```.

The output files are downloaded locally in ```outputs/INFER_${MODEL_NAME}_${MODEL_VERSION}``` folder.

---
To visualize some detected containers, run

``` python predictions.py  --name ${MODEL_NAME} --version ${MODEL_VERSION} --device cpu```


## Single Image Prediction
Firstly, make sure you have the the model weights with ```${MODEL_NAME}``` and ```${MODEL_VERSION}``` locally in 
```outputs/${MODEL_NAME}_${MODEL_VERSION}``` folder.

To detect containers on a given image, run

```python predictions.py --name ${MODEL_NAME} --version ${MODEL_VERSION} --image ${PATH}``` 

The output image is stored in the same folder as ```${ORIGINAL_IMAGE_NAME}:out```
## Evaluation Metrics

To compute and store evaluation metrics, run
```shell
python evaluate.py
```
We use the following metrics:

1. **Average Precision (AP)** with the following IoU thresholds: 0.5, 0.75 and [0.5-0.95]
2. **Average Recall (AR)**  with the following IoU thresholds: 0.5, 0.75 and [0.5-0.95]

We calculate AP and AR for the following list of maximum detections: **1**, **10** and **100**.

We calculate AP and AR for the following list of bounding box areas: **small**, **medium** and **large** , where

- **small** = [0 ** 2, 32 ** 2], i.e. objects that have area between 0 pixels and 32*32 (1024) pixels 
- **medium** = [32 ** 2, 96 ** 2],i.e. objects that have area between 32 * 32 pixels and 96 * 96 (9216) pixels
- **large** = [96 ** 2, 1e5 ** 2]], i.e. objects that have area between 96 * 96 pixels and 1e5 * 1e5 (1e10) pixels

We evaluate both **bounding boxes** results and **segmentation** results.

### More notes on the evaluation metrics


####Precision 
Precision measures how many of the predictions that the model made were actually correct.
For each bounding box, we measure the overlap between the predicted bounding box and the ground truth bounding box, 
i.e. intersection over union (IoU). Precision is calculated based on the IoU threshold.

####Recall
Recall measures how well the model finds all the positives out of all the predictions it makes. 
For example, we can find 80% of the possible positive cases in the top K predictions. Recall is also calculated based on the IoU threshold. 

---

In COCO, mAP=AP and mAR=AR.

AP is calculated by taking the area under the precison-recall curve. If we have multiple IoU thresholds, we average the 
AP corresponding to each threshold.

When we compute AR, we again take a set of IoU values  [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]. 
Then, for each category, compute the recall at detecting that category at the specific IOU threshold. Then consider two 
cases
1. a mean average recall at a specific IOU value
2. a mean average recall for all IOU values

Both cases can be considered over a varying amount of maximum detections, 1, 10, 100.

For 1, we compute the recall for each category, and then take the average of those values over all classes.

For 2, we average the recall at each of the IoU threshold, and then take the average of this average for each class. This is our AR@{D} for D detections.


## Visualizations of Heavy Objects

The model detects construction containers in images that are provided by the panorama car. This car 
which drives through the city and takes panoramic images of the streets.
Subsequently, these images are uploaded to the [panorama API](/link_to_api).

Given a day D when the panoramic car drove through the city, we can visualize both the trajectory of the car and the 
detected containers on this trajectory.
To create the map, navigate to the [visualizations](./visualizations) folder and run
```shell
python -m visualisations.daily_trajectory
```
The day to plot can be configured in the script.
The HTML map is created in the same folder. 

---
## Self notes: Creating a new dataset via Data Labelling
We first finish annotations
Download images
Merge all folders into one
Download the annotation from AzureML

Run clustering notebook.
Now we have 3 folders: train, validation, test

Now we need to resize all images to 2000x4000 aka run correct_faulty_panoramas()

This will generate folder with upscaled images.
These images must be manually moved and replace the downscaled ones in the 3 folders that result from the notebook.

Then we apply the data format converter aka run
The converter on the output of the correct_faulty_panoramas()


These output files we can rename and upload to Azure.