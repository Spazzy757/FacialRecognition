from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient as Client
from django.test import TestCase
from api.models import File, Dog
import os
import re


class DogAPIUploadTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.dog = Dog.objects.create(
            name='Fluffy',
            labels=['terrier']
        )
        self.file = File.objects.create(
            file=SimpleUploadedFile(
                name='test_image.jpg',
                content=open('api/tests/test_image.jpg', 'rb').read(),
                content_type='image/jpeg'
            ),
            remark='test'
        )

    def test_post(self):
        data = {
            'name': 'Spot',
            'labels': ['labrador']
        }
        result = self.client.post('/api/dog/', data=data)
        self.assertTrue(result.status_code == 201,
                        msg="Failed to create Dog via POST")

    def test_add_file_to_dog(self):
        # set_clean_dataset()
        data = {
            'file': self.file.id,
            'dog': self.dog.id
        }
        result = self.client.post('/api/add-file-to-dog/', data=data)
        self.assertTrue(result.status_code == 201,
                        msg="Failed add file to dog")
        dog_check = Dog.objects.get(id=self.dog.id)
        self.assertTrue(dog_check.generated_labels.count() > 0,
                        msg="Failed to run analysis")

    def tearDown(self):
        def purge(directory, pattern):
            for f in os.listdir(directory):
                if re.search(pattern, f):
                    os.remove(os.path.join(directory, f))
        purge('media/images', 'test_image*')
