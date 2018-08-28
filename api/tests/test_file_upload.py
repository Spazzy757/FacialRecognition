from rest_framework.test import APIClient as Client
from django.test import TestCase
from api.models import File


class FileUploadTestCase(TestCase):
    def setUp(self):
        self.client = Client()

    def test_post(self):
        with open('api/tests/test_image.jpg', 'rb') as fp:
            self.client.post(
                '/api/upload/',
                {
                    'remark': 'mum',
                    'file': fp
                }
            )
            self.assertTrue(File.objects.count() == 1)

    def test_clean_data(self):
        # set_clean_dataset()
        pass
