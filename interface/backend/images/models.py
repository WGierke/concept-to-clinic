import glob

import dicom
from django.db import models
from django.utils._os import safe_join


class ImageSeries(models.Model):
    """
    Model representing a certain image series
    """
    patient_id = models.CharField(max_length=64)

    series_instance_uid = models.CharField(max_length=256)

    uri = models.CharField(max_length=512)

    acquisition_date = models.CharField(max_length=64)

    age = models.IntegerField()

    sex = models.CharField(max_length=6)

    peak_kilovoltage = models.IntegerField()

    # milliampere = models.IntegerField()

    def get_or_create(uri):
        """
        Return the ImageSeries instance with the same PatientID and SeriesInstanceUID as the DICOM images in the
        given directory. If none exists so far, create one.
        Return a tuple of (ImageSeries, created), where created is a boolean specifying whether the object was created.

        Args:
            uri (str): absolute URI to a directory with DICOM images of a patient

        Returns:
            (ImageSeries, bool): the looked up ImageSeries instance and whether it had to be created
        """
        file_ = glob.glob1(uri, '*.dcm')[0]
        plan = dicom.read_file(safe_join(uri, file_))
        patient_id = plan.PatientID
        series_instance_uid = plan.SeriesInstanceUID
        acquisition_date = plan.AcquisitionDate
        age = plan.PatientAge
        sex = plan.PatientSex
        kvp = plan.KVP

        return ImageSeries.objects.get_or_create(
            patient_id=patient_id,
            series_instance_uid=series_instance_uid,
            uri=uri, acquisition_date=acquisition_date,
            age=age, sex=sex, peak_kilovoltage=kvp)


class ImageLocation(models.Model):
    """
    Model representing a certain voxel location on certain image
    """
    series = models.ForeignKey(ImageSeries, on_delete=models.CASCADE)

    x = models.PositiveSmallIntegerField(help_text='Voxel index for X axis, zero-index, from top left')

    y = models.PositiveSmallIntegerField(help_text='Voxel index for Y axis, zero-index, from top left')

    z = models.PositiveSmallIntegerField(help_text='Slice index for Z axis, zero-index')
