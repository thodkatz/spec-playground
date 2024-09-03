from bioimageio.spec.model import v0_4
import numpy as np
from myspec.dummy_network import DummyNetwork
import tempfile
from pathlib import Path
import torch
from datetime import datetime


dummy_network_module = "../dummy_network.py"

def get_bioimage_model_v4():
    output_test_tensor = np.arange(1 * 2 * 10 * 10, dtype="float32").reshape(1, 2, 10, 10)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as output_test_tensor_file:
        np.save(output_test_tensor_file.name, output_test_tensor)

    input_test_tensor = np.arange(1 * 2 * 10 * 10, dtype="float32").reshape(1, 2, 10, 10)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as input_test_tensor_file:
        np.save(input_test_tensor_file.name, input_test_tensor)

    dummy_model = DummyNetwork()
    with tempfile.NamedTemporaryFile(suffix=".pts", delete=False) as weights_file:
        torch.save(dummy_model.state_dict(), weights_file.name)

    input = v0_4.InputTensorDescr(
        name=v0_4.TensorName("input"), description="", axes="bcxy", shape=[1, 2, 10, 10], data_type="float32"
    )

    output = v0_4.OutputTensorDescr(
        name=v0_4.TensorName("output"), description="", axes="bcxy", shape=[1, 2, 10, 10], data_type="float32"
    )

    model_descr = v0_4.ModelDescr(
        name="mocked v4 model",
        authors=[v0_4.Author(name="me")],
        cite=[v0_4.CiteEntry(text="for model training see my paper", url=v0_4.HttpUrl("https://doi.org/10.1234something"))],
        description="",
        inputs=[input],
        outputs=[output],
        documentation=v0_4.HttpUrl("https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/README.md"),
        license="MIT",
        test_inputs=[Path(input_test_tensor_file.name)],
        test_outputs=[Path(output_test_tensor_file.name)],
        timestamp=v0_4.Datetime(root=datetime.now()),
        # weights=v0_4.WeightsDescr(
        #     pytorch_state_dict=v0_4.PytorchStateDictWeightsDescr(
        #         source=Path(weights_file.name),
        #         architecture=f"{dummy_network_module}:{DummyNetwork.__name__}",
        #         architecture_sha256="cb1e8501cca44a63f3acd91cf35da121063156d4bd8213d466614f6a84bbfbfa"
        #     )
        # ),
        weights=v0_4.WeightsDescr(
            pytorch_state_dict=v0_4.PytorchStateDictWeightsDescr(
                source=Path(weights_file.name),
                architecture="myspec.dummy_network.DummyNetwork",
            )
        )
    )
    return model_descr