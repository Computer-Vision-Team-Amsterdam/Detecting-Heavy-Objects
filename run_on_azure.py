from azureml.core import ScriptRunConfig, Experiment, ComputeTarget, Dataset, Environment, Workspace, Datastore
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


    base_name = "test-heavy"
    image = "amlacri1xfnr.azurecr.io/test-heavy"
    env = Environment(base_name)
    env.docker.base_image = image
    env.inferencing_stack_version = "latest"
    env.python.user_managed_dependencies = True


    docker_config = DockerConfiguration(use_docker=True)
    # Get datastore path
    default_ds: Datastore = ws.get_default_datastore()

    # Register a dataset for the input data
    data_set = Dataset.File.from_files(path=(default_ds, 'data/train/'), validate=False)

    # TODO: check if create_new_version should stay True since it's the same data everytime for now
    container_data = data_set.register(workspace=ws,
                                       name='container-data',
                                       description='container-data')
    dataset = Dataset.get_by_name(ws, 'container-data')

    compute_target = ComputeTarget(ws, "DianaComputeInstance")

    # Create a script config for the experiment
    experiment_name = "train_dummy_detectron2"

    script_config = ScriptRunConfig(source_directory=".",
                                    script='training.py',
                                    arguments=[],
                                    environment=env,
                                    compute_target=compute_target,
                                    docker_runtime_config=docker_config
                                    )
    # Submit the experiment

    experiment = Experiment(workspace=ws, name=experiment_name)
    run = experiment.submit(config=script_config)
    RunDetails(run)

    # add if statement for registering
    run.wait_for_completion(show_output=True)
    # run.register_model("test_model", f"outputs/exp0_{experiment_name}/weights/best.pt")