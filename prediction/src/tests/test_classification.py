from os import path

import pylidc as pl
from config import Config
from tqdm import tqdm

from ..algorithms.classify import trained_model


def get_slice_number_from_z_value(scan, z_value):
    images = scan.load_all_dicom_images()
    start_z = images[0].ImagePositionPatient[-1]
    return int((z_value - start_z) / scan.slice_thickness)


def test_classify_dicom(dicom_paths, nodule_locations, model_path):
    scans = pl.query(pl.Scan).all()
    MALICIOUS = 5
    n = 0
    error = 0

    for scan in tqdm(scans[:1]):
        for a in scan.annotations:
            if a.malignancy == MALICIOUS:
                dir_ = '{}/{}/{}'.format(path.join(Config.FULL_DICOM_PATHS, scan.patient_id), scan.study_instance_uid,
                                         scan.series_instance_uid)
                centroid = list(a.centroid())
                centroid[-1] = get_slice_number_from_z_value(scan, centroid[-1])
                centroid_dict = {'x': centroid[0], 'y': centroid[1], 'z': centroid[2]}
                predicted = trained_model.predict(dir_, [centroid_dict], model_path)
                error += (predicted - 1) ** 2
                n += 1
        print(error / n)
