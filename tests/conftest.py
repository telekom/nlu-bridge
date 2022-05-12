import pytest

from testing_data import TrainingDataset


@pytest.fixture
def train_data():
    return TrainingDataset()
