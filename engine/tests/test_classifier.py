from engine.recognition import NameClassifier
from django.test import TestCase
import os


class CapacityImportTestCase(TestCase):

    def test_recognition(self):
        recog = NameClassifier('engine/images')
        full_file_path = os.path.join("engine/tests/data/", 'unknown.py')

        print("Looking for faces in {}".format('unknown'))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier
        # model instance
        predictions = recog.predict(full_file_path,
                                    model_path="trained_knn_model.clf")
        import ipdb; ipdb.set_trace()
        # # Print results on the console
        # for name, (top, right, bottom, left) in predictions:
        #     print("- Found {} at ({}, {})".format(name, left, top))

