from django.db import models


# Create your models here.
class Image(models.Model):
    name = models.CharField(max_length=256)
    description = models.TextField(max_length=512)

    @property
    def get_file_path(self):
        return ''.format(self.name)
