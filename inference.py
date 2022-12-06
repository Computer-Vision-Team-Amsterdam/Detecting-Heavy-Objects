"""This module contains functionality to load a model and run predictions that can be
incorporated into the Azure batch processing pipeline"""
import argparse
import logging
import os
from pathlib import Path
from typing import Union

from detectron2.config import CfgNode, get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import inference_on_dataset

from utils.azure_storage import BaseAzureClient, StorageAzureClient
from configs.config_parser import arg_parser
from evaluation import CustomCOCOEvaluator  # type:ignore
from utils_coco import ExperimentConfig, register_dataset
from utils.date import get_start_date

azClient = BaseAzureClient()


logging.basicConfig(level=logging.INFO)


def setup_cfg(config_file: Union[Path, str]) -> CfgNode:
    """
    Reads the model and inference settings from the config file
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    return cfg


def init_inference(flags: argparse.Namespace) -> CfgNode:
    """
    Initializes the trained model
    :param flags: console arguments
    Returns the configuration object
    """
    config_file = Path(flags.config)

    cfg = setup_cfg(config_file)

    cfg.MODEL.WEIGHTS = flags.weights
    print(f"using weights: {cfg.MODEL.WEIGHTS}")
    cfg.MODEL.DEVICE = flags.device

    return cfg


def evaluate_model(flags: argparse.Namespace, expCfg: ExperimentConfig) -> None:
    """
    This method calculates evaluation metrics for the trained model.
    :param expCfg: experiment we evaluate
    """
    # pylint: disable=too-many-function-args
    """
    note to self: pylint wrongly detects that build_detection_test_loader is called with 
    too many positional args but the method has a decorator with which we can use the
    constructor from another method i.e. _test_loader_from_config
    """

    register_dataset(expCfg)
    cfg = init_inference(flags)
    cfg.MODEL.WEIGHTS = (
        flags.weights
    )  # replace with get_Azure_model() if no local weights

    cfg.OUTPUT_DIR = flags.output_path

    predictor = DefaultPredictor(cfg)

    logging.info(f"Loaded model weights from {cfg.MODEL.WEIGHTS}.")

    # run_name = datetime.now().strftime("%b-%d-%H:%M")
    # output_dir = f"{cfg.OUTPUT_DIR}/{run_name}"
    output_dir = f"{cfg.OUTPUT_DIR}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluator = CustomCOCOEvaluator(
        f"{expCfg.dataset_name}_{flags.subset}",
        output_dir=output_dir,
        tasks=("bbox", "segm"),
    )
    loader = build_detection_test_loader(
        cfg, f"{expCfg.dataset_name}_{flags.subset}", mapper=None
    )
    print(inference_on_dataset(predictor.model, loader, evaluator))


if __name__ == "__main__":
    flags = arg_parser()

    start_date_dag, start_date_dag_ymd = get_start_date(flags.subset)

    input_path = Path(flags.data_folder, start_date_dag_ymd)
    if not input_path.exists():
        input_path.mkdir(exist_ok=True, parents=True)

    # Download images from storage account
    saClient = StorageAzureClient(secret_key="data-storage-account-url")
    blobs = saClient.list_container_content(cname="blurred", blob_prefix=start_date_dag)
    for blob in blobs:
        filename = blob.split("/")[-1]  # only get file name, without prefix
        saClient.download_blob(
            cname="blurred",
            blob_name=blob,
            local_file_path=f"{input_path}/{filename}",
        )

    flags.subset = start_date_dag_ymd
    experimentConfig = ExperimentConfig(
        dataset_name=flags.dataset_name,
        subset=flags.subset,
        data_format=flags.data_format,
        data_folder=flags.data_folder,
    )
    evaluate_model(flags, experimentConfig)

    # Upload detection files to postgres
    for file in os.listdir(flags.output_path):
        saClient.upload_blob(
            cname="detections",
            blob_name=f"{start_date_dag}/{file}",
            local_file_path=f"{flags.output_path}/{file}",
        )
