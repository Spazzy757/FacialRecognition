from engine.recognition import BreedClassifier, set_clean_dataset
from django.test import TestCase
from engine.models import Image
import os


class CapacityImportTestCase(TestCase):

    def setUp(self):
        Image.objects.create(name='lisa')
        Image.objects.create(name='neil')
        Image.objects.create(name='nana')

    def test_breed_classifier(self):
        classifier = BreedClassifier()
        result = classifier.classify('media/test_images/bear2.jpg')


    def test_clean_data(self):
        pass
        # set_clean_dataset()
