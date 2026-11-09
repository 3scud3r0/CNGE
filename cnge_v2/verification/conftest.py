import pytest
import torch


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(41)
    torch.set_num_threads(1)
