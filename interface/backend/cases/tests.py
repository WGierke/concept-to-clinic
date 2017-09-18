import json

from backend.cases.factories import (
    CandidateFactory,
    CaseFactory,
    NoduleFactory
)
from backend.cases.models import Nodule
from django.test import TestCase
from django.urls import reverse


class SmokeTest(TestCase):
    def test_create_case(self):
        case = CaseFactory()
        self.assertIsNotNone(case.series)

    def test_create_candidate(self):
        candidate = CandidateFactory()

        # check p(concerning)
        self.assertGreater(candidate.probability_concerning, 0.0)
        self.assertLess(candidate.probability_concerning, 1.0)

        # check the centroid location
        self.assertIsInstance(candidate.centroid.x, int)
        self.assertIsInstance(candidate.centroid.y, int)
        self.assertIsInstance(candidate.centroid.z, int)

    def test_create_nodule(self):
        nodule = NoduleFactory()

        # check the centroid location
        self.assertIsInstance(nodule.centroid.x, int)
        self.assertIsInstance(nodule.centroid.y, int)
        self.assertIsInstance(nodule.centroid.z, int)

    def test_update_nodule_lung_orientation(self):
        nodule = NoduleFactory()
        url = reverse('nodule-update', kwargs={'nodule_id': nodule.id})

        self.assertEquals(nodule.lung_orientation, Nodule.LungOrientation.NONE.value)

        self.client.post(url, json.dumps({'lung_orientation': 'LEFT'}), 'application/json')
        nodule.refresh_from_db()
        self.assertEquals(nodule.lung_orientation, Nodule.LungOrientation.LEFT.value)

        self.client.post(url, json.dumps({'lung_orientation': 'RIGHT'}), 'application/json')
        nodule.refresh_from_db()
        self.assertEquals(nodule.lung_orientation, Nodule.LungOrientation.RIGHT.value)

        self.client.post(url, json.dumps({'lung_orientation': 'NONE'}), 'application/json')
        nodule.refresh_from_db()
        self.assertEquals(nodule.lung_orientation, Nodule.LungOrientation.NONE.value)
