from src.algorithms.segment.src.data_generation import prepare_training_data_cubes
from src.algorithms.segment.src.training import train


def test_3d_unet():
    prepare_training_data_cubes()
    train()


