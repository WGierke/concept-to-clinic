import glob
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pylidc as pl


def get_dicom_paths():
    """Return DICOM paths to all LIDC directories
    e.g. ['../images_full/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/' \
          '1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192']"""
    return glob.glob(os.path.join(os.getcwd(), "tests/assets/test_image_data/full2/LIDC-IDRI-*/**/**"))
    return glob.glob("../images_full/LIDC-IDRI-*/**/**")


def prepare_training_data_cubes():
    """Save cubes of nodules and nodule segmentations.
    Iterate over all local LIDC images, fetch the annotations and save one cube of CT image including the annotation
    and one cube with the segmented annotation in a binary mask."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    assets_dir = os.path.abspath(os.path.join(current_dir, '../assets'))
    dicom_paths = get_dicom_paths()

    for path in dicom_paths:
        directories = path.split('/')
        print(directories)
        lidc_id = directories[2]
        lidc_id = directories[10]
        print("Handling ", lidc_id)
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == lidc_id).first()
        for annotation in scan.annotations:
            input_exists = os.path.isfile(os.path.join(assets_dir, "annotation_{}_input.npy".format(annotation.id)))
            output_exists = os.path.isfile(os.path.join(assets_dir, "annotation_{}_output.npy".format(annotation.id)))
            if not input_exists and not output_exists:
                cube_volume, cube_segmentation = annotation.uniform_cubic_resample(side_length=63)
                np.save(os.path.join(assets_dir, "annotation_{}_input.npy".format(annotation.id)), cube_volume)
                np.save(os.path.join(assets_dir, "annotation_{}_output.npy".format(annotation.id)), cube_segmentation)
