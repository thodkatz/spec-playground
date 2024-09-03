import xarray as xr
import numpy as np
import torch.nn as nn

dummy_network_in_file = xr.DataArray(np.arange(2 * 10 * 10).reshape(1, 2, 10, 10), dims=["batch", "channel", "x", "y"])


class DummyNetwork(nn.Module):
    def forward(self, *args):
        return dummy_network_in_file
