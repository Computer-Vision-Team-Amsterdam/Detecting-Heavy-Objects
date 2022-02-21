"""
This module contains functionality to run a training script on Azure.
"""
from azureml.core import (
    ComputeTarget,
    Dataset,
    Datastore,
    Environment,
    Experiment,
    ScriptRunConfig,
    Workspace,
)
from tqdm import trange

import json

from utils import handle_hyperparameters

EXPERIMENT_NAME = "hyperparam_search"


if __name__ == "__main__":
    ws = Workspace.from_config()

    env = Environment.from_dockerfile("cuda_env_container", "Dockerfile")

    # Get datastore path
    default_ds: Datastore = ws.get_default_datastore()

    dataset = Dataset.get_by_name(ws, "container-data")

    mounted_dataset = dataset.as_mount(path_on_compute="data/")

    compute_target = ComputeTarget(ws, "detectron2")

    nr_experiments = handle_hyperparameters("configs/hyperparameter-search.yaml")

    for exp in trange(1):
        CONFIG = f"configs/temp_{exp}.yaml"
        # Create a script config for the experiment
        script_config = ScriptRunConfig(
            source_directory=".",
            script="training.py",
            arguments=["--dataset", mounted_dataset, "--config", CONFIG],
            environment=env,
            compute_target=compute_target,
        )
        # Submit the experiment
        experiment = Experiment(workspace=ws, name=EXPERIMENT_NAME)
        run = experiment.submit(config=script_config)
        run.display_name = f"temp_{exp}"

        output_list = [json.loads(line) for line in open('outputs/metrics.json', 'r')]
        keys = output_list[0].keys()
        for key in keys:
            for output in output_list:
                run.log(name=key, value=output[key])

        # add if statement for registering
        run.wait_for_completion(show_output=True)

        # run.register_model("test_model", f"outputs/")
