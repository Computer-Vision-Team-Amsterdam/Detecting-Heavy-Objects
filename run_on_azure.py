"""
This module contains functionality to run a training script on Azure.
"""

import azureml._restclient.snapshots_client
from azureml.exceptions import WebserviceException

azureml._restclient.snapshots_client.SNAPSHOT_MAX_SIZE_BYTES = 1000000000

from configs.config_parser import arg_parser
from azureml.core import (
    ComputeTarget,
    Dataset,
    Datastore,
    Environment,
    Experiment,
    ScriptRunConfig,
    Workspace, Model,
)

EXPERIMENT_NAME = "detectron_2000x4000" # all used images have resolution 2000x4000

ws = Workspace.from_config()
env = Environment.from_dockerfile("cuda_env_container", "Dockerfile")
default_ds: Datastore = ws.get_default_datastore()
dataset = Dataset.get_by_name(ws, "blurred-container-data")
mounted_dataset = dataset.as_mount(path_on_compute="data/")
compute_target = ComputeTarget(ws, "quick-gpu")
experiment = Experiment(workspace=ws, name=EXPERIMENT_NAME)

flags = arg_parser()

# check if model already exists. Used when creating the name of the output folder.
try:
    model = Model(ws, f"{flags.name}")
    if flags.train:
        flags.version = model.version + 1
except WebserviceException:
    flags.version = 1

args = {}
for arg in vars(flags):
    if flags.train is False and arg == "train":
        continue
    if flags.inference is False and arg == "inference":
        continue
    args[f"--{arg}"] = getattr(flags, arg)


args["--dataset"] = mounted_dataset
args = [[k, v] for (k, v) in args.items()]
args = [val for sublist in args for val in sublist]  # flatten


if flags.train:
    script_config = ScriptRunConfig(
        source_directory=".",
        script="training.py",
        arguments=args,
        environment=env,
        compute_target=compute_target,
    )
    run = experiment.submit(config=script_config)
    run.wait_for_completion(show_output=True)

    run.register_model(flags.name, f"outputs/TRAIN_{flags.name}_{flags.version}/model_final.pth")
    run.download_files(prefix="outputs")


if flags.inference:
    model = Model.get_model_path(model_name=f"{flags.name}", version=flags.version, _workspace=ws)  # latest version

    script_config = ScriptRunConfig(
        source_directory=".",
        script="inference.py",
        arguments=args,
        environment=env,
        compute_target=compute_target,
    )
    run = experiment.submit(config=script_config)
    run.wait_for_completion(show_output=False)
    run.download_files(prefix="outputs")