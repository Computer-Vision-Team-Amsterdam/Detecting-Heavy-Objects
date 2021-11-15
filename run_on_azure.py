# The download location can be retrieved from argument values
import sys

# The download location can also be retrieved from input_datasets of the run context.
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
from azureml.core.runconfig import DockerConfiguration
from azureml.widgets import RunDetails

if __name__ == "__main__":
    ws = Workspace.from_config()

    """
    # Create a Python environment for the experiment
    env = Environment.from_conda_specification(name="detectron2",
                                                       file_path="detectron2.yml")

    # for some reason the python version is not automatically retrieved from yml file
    env.python.conda_dependencies.set_python_version("3.8.5")
    """

    env = Environment.from_dockerfile("cuda_env_container", "Dockerfile")

    # env.inferencing_stack_version = "latest"
    # env.python.user_managed_dependencies = False

    # Get datastore path
    default_ds: Datastore = ws.get_default_datastore()

    # Register a dataset for the input data
    data_set = Dataset.File.from_files(
        path=(default_ds, "UI/11-03-2021_070053_UTC/data/train/"), validate=False
    )

    # TODO: check if create_new_version should stay True since it's the same data everytime for now
    """
    container_data = data_set.register(workspace=ws,
                                       name='container-data',
                                       description='container-data')
    """
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
    # run.register_model("test_model", f"outputs/exp0_{experiment_name}/weights/best.pt")
