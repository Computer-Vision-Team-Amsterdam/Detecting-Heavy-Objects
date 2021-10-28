from contextlib import contextmanager

from azureml.core import Environment, Experiment, Model, ScriptRunConfig, Workspace
from azureml.core.compute import ComputeInstance
from azureml.core.compute_target import ComputeTargetException
from azureml.data import OutputFileDatasetConfig
from azureml.core.model import InferenceConfig
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep
from azureml.core.webservice import LocalWebservice


@contextmanager
def provision_compute(ws, name, vm_size):
    try:
        _compute = ComputeInstance(workspace=ws, name=name)
        if _compute.get_status().state.upper() != "RUNNING":
            _compute.start(wait_for_completion=True)
    except ComputeTargetException:
        compute_config = ComputeInstance.provisioning_configuration(
            vm_size=vm_size, ssh_public_access=False
        )
        _compute = ComputeInstance.create(ws, name, compute_config)
        _compute.wait_for_completion(show_output=True)
    try:
        yield _compute
    finally:
        # _compute.delete()
        pass


if __name__ == "__main__":
    # Set up workspace
    base_name = "test-heavy"
    # vm_size = "Standard_NC6s_v3"
    vm_size = "Standard_DS3_v2"
    image = "amlacri1xfnr.azurecr.io/test-heavy"
    ws = Workspace.from_config(".azure/config.json")

    # Retrieve model
    model = Model(ws, "detectron2")

    # Retrieve dataset
    # dataset = ws.datasets[base_name].as_named_input("input").as_mount("/opt/pysetup/mnt")
    dataset = ws.datasets[base_name]

    # Set up container environment
    env = Environment(base_name)
    env.docker.base_image = image
    env.inferencing_stack_version = "latest"
    env.python.user_managed_dependencies = True

    # exp = Experiment(ws, base_name)

    output_dir = OutputFileDatasetConfig(name='inferences')

    with provision_compute(ws, base_name, vm_size) as compute:
        parallel_run_config = ParallelRunConfig(
            source_directory='scripts',
            entry_script="run_inference.py",
            mini_batch_size="5",
            error_threshold=10,
            output_action="append_row",
            environment=env,
            compute_target=compute,
            node_count=1,
        )

        parallel_run_step = ParallelRunStep(
            name='batch-score',
            parallel_run_config=parallel_run_config,
            inputs=[dataset.as_named_input("input")],
            output=output_dir,
            arguments=[],
            allow_reuse=True,
        )

        pipeline = Pipeline(workspace=ws, steps=[parallel_run_step])

        pipeline_run = Experiment(ws, base_name).submit(pipeline)
        pipeline_run.wait_for_completion(show_output=True)

    prediction_run = next(pipeline_run.get_children())
    prediction_output = prediction_run.get_output_data('inferences')
    prediction_output.download(local_path='results')

    # inference_config = InferenceConfig(
    #     environment=env,
    #     source_directory=".",
    #     entry_script="scripts/run_inference.py"
    # )
    # deployment_config = LocalWebservice.deploy_configuration(port=6789)
    # service = Model.deploy(
    #     ws,
    #     "detectron2service",
    #     [model],
    #     inference_config,
    #     deployment_config,
    #     overwrite=True,
    # )
    # service.wait_for_deployment(show_output=True)

    # with compute(ws, base_name, vm_size) as gpu_compute:
    #     config = ScriptRunConfig(
    #         source_directory=".",
    #         script="scripts/run_inference.py",
    #         compute_target=gpu_compute,
    #         environment=env,
    #         arguments=[dataset],
    #     )
    #
    #     # submit script to AML
    #     run = exp.submit(config)
    #     print(run.get_portal_url()) # link to ml.azure.com
    #     run.wait_for_completion(show_output=True)
