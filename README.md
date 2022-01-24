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
python training.py
```
The output files are stored according to the [training configuration file](./configs/container_detection_train.yaml).


To train the model on GPU, run 
```shell
python run_on_azure.py --TODO add flag to change devide from cpu to cuda:0
```

The output files are stored on Azure in the *outputs* folder and contain the following:

- *model_final.pth* -- trained pytorch model
- *metrics.json* -- metrics used for plotting by the tensorboard
- *events.out.tfevents* -- output summary used for plotting by tensorboard
- *last_checkpoint* -- stores reference to name of the last trained model.



Once you have a trained model, you can visualize the training loss curves, accuracies and other metrics by running:
```shell
python -m tensorboard.main --logdir $PWD/output
```

## Model Inference

To perform inference on your local machine, run
```shell
python inference.py
```
If there is no trained model stored locally, this command downloads a trained model from Azure and stores it in 
the *output*. folder. To download a specific model, run:
```shell
python inference.py --name MODEL_NAME --version VERSION 
```


To perform inference on GPU, run
```shell
python inference.py --TODO add flag to change devide from cpu to cuda:0
```
The output files are as follows:
- *container_val_coco_format.json* -- automatically generated COCO file for Detectron2. It contains metadata of the 
dataset used at inference time 
- *coco_instances_results.json* -- predictions of all instances
- *instances_predictions.pth* -- ? 
- *eval_metrics.txt* -- stores evaluation metrics described below

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
python daily_trajectory.py
```
The day to plot can be configured in the script.
The HTML map is created in the same folder. 