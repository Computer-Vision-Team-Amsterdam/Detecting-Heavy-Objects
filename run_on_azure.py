from azureml.core import (
    ComputeTarget,
    Dataset,
    Datastore,
    Environment,
    Experiment,
    Run,
    ScriptRunConfig,
    Workspace,
)
from azureml.widgets import RunDetails

if __name__ == "__main__":
    ws = Workspace.from_config()

    env = Environment.from_dockerfile("cuda_env_container", "Dockerfile")

    # Get datastore path
    default_ds: Datastore = ws.get_default_datastore()

    # Register a dataset for the input data
    data_set = Dataset.File.from_files(
        path=(default_ds, "UI/11-03-2021_070053_UTC/data/train/"), validate=False
    )

    dataset = Dataset.get_by_name(ws, "container-data")

    downloaded = dataset.as_download(path_on_compute="data/")

    compute_target = ComputeTarget(ws, "detectron2")

    # Create a script config for the experiment
    experiment_name = "train_dummy_detectron2"

    script_config = ScriptRunConfig(
        source_directory=".",
        script="training.py",
        arguments=["--dataset", downloaded],
        environment=env,
        compute_target=compute_target,
    )
    # Submit the experiment
    experiment = Experiment(workspace=ws, name=experiment_name)
    run = experiment.submit(config=script_config)
    # RunDetails(run)

    # add if statement for registering
    run.wait_for_completion(show_output=True)
    # run.register_model("test_model", f"outputs/model_final.pth")
