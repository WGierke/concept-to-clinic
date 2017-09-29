import glob
import os

import numpy as np
from keras.callbacks import ModelCheckpoint

from .model import unet_model_3d


def train():
    """Load the training cubes in the asset folder and train the 3D U-Net on them"""
    CUBE_IMAGE_SHAPE = (64, 64, 64, 1)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    assets_dir = os.path.abspath(os.path.join(current_dir, '../assets'))
    inputs = glob.glob(os.path.join(assets_dir, "annotation_*_input.npy"))
    outputs = glob.glob(os.path.join(assets_dir, "annotation_*_output.npy"))
    input_data = np.ndarray((len(inputs), *CUBE_IMAGE_SHAPE))
    output_data = np.ndarray((len(inputs), *CUBE_IMAGE_SHAPE))

    for index, (in_file, out_file) in enumerate(zip(inputs, outputs)):
        input_cube = np.load(in_file)
        output_cube = np.load(out_file)

        # Expand dimensions
        input_cube = np.expand_dims(input_cube, axis=0)
        output_cube = np.expand_dims(output_cube, axis=0)
        # Trailing channel is necessary for Tensorflow
        input_cube = input_cube.reshape(*CUBE_IMAGE_SHAPE)
        output_cube = output_cube.reshape(*CUBE_IMAGE_SHAPE)

        input_data[index, :, :, :] = input_cube
        output_data[index, :, :, :] = output_cube

    model = unet_model_3d(CUBE_IMAGE_SHAPE, downsize_filters_factor=32)
    model_checkpoint = ModelCheckpoint(os.path.join(assets_dir, 'best.hdf5'), monitor='loss', verbose=1, save_best_only=True)
    print(input_data.shape)
    print(output_data.shape)
    model.fit(input_data, output_data, callbacks=[model_checkpoint], epochs=10)
