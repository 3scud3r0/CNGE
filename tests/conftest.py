import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--long", action="store_true", help="run 50,000-step validations")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--long"):
        return
    for item in items:
        if "long" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="use --long for 50,000-step validation"))


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(17)
    torch.set_num_threads(1)
