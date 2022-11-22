"""
This module contains functionality to run a training script on Azure.

python run_train_on_azure.py --name test-on-as --version 1 --subset train -data_folder data_sample_2 --dataset data-sample-2

"""
from typing import Any, Dict, List

import azureml._restclient.snapshots_client
from azureml.exceptions import WebserviceException

azureml._restclient.snapshots_client.SNAPSHOT_MAX_SIZE_BYTES = 1000000000

from azureml.core import (
    ComputeTarget,
    Dataset,
    Environment,
    Experiment,
    Model,
    ScriptRunConfig,
    Workspace,
)

from configs.config_parser import arg_parser

EXPERIMENT_NAME = "test_with_data_sample"


ws = Workspace.from_config()
env = Environment.from_dockerfile("cuda_env_container", "Dockerfile")

dataset = Dataset.get_by_name(ws, "container-project-dataset-2")

mounted_dataset = dataset.as_mount(path_on_compute="container-project-dataset/")
compute_target = ComputeTarget(ws, "container-model-gpu")
experiment = Experiment(workspace=ws, name=EXPERIMENT_NAME)

flags = arg_parser()

# check if model already exists. Used when creating the name of the output folder.
try:
    model = Model(ws, f"{flags.name}")
    flags.version = model.version + 1
except WebserviceException:
    flags.version = 1

args: Dict[str, Any] = {}
for arg in vars(flags):
    args[f"--{arg}"] = getattr(flags, arg)

args["--dataset"] = mounted_dataset
args_list: List[List[Any]] = [[k, v] for (k, v) in args.items()]
args_flattened: List[str] = [val for sublist in args_list for val in sublist]  # flatten

script_config = ScriptRunConfig(
    source_directory=".",
    script="training.py",
    arguments=args_flattened,
    environment=env,
    compute_target=compute_target,
)
run = experiment.submit(config=script_config)
run.wait_for_completion(show_output=True)


run.register_model(
    flags.name, f"outputs/TRAIN_{flags.name}_{flags.version}/model_final.pth"
)
run.download_files(prefix="outputs")
