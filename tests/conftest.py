import pytest
from testing_data import TrainingDataset


@pytest.fixture
def train_data():
    """Provide testing_data.TrainingDataset as fixture."""
    return TrainingDataset()
