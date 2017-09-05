from backend.cases.models import (
    Case,
    Candidate,
    Nodule,
)
from backend.images.models import (
    ImageSeries,
    ImageLocation
)
from rest_framework import serializers


class ImageSeriesSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ImageSeries
        fields = '__all__'


class ImageLocationSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageLocation
        fields = '__all__'


class CaseSerializer(serializers.HyperlinkedModelSerializer):
    series = ImageSeriesSerializer()

    class Meta:
        model = Case
        fields = ('series',)
        read_only_fields = ('created',)

    def create(self, validated_data):
        series_data = validated_data.pop('series')
        image_series = ImageSeries.objects.create(**series_data)
        case = Case.objects.create(series_id=image_series.id, **validated_data)
        return case


class CandidateSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Candidate
        fields = '__all__'
        read_only_fields = ('created',)

    centroid = ImageLocationSerializer()


class NoduleSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Nodule
        fields = '__all__'
        read_only_fields = ('created',)

    centroid = ImageLocationSerializer()
