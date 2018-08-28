from django.contrib.postgres.fields import ArrayField
from django.db import models


class File(models.Model):
    file = models.ImageField(blank=False, null=False)
    remark = models.CharField(max_length=20)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return "{}".format(self.file)


class Dog(models.Model):
    name = models.CharField(max_length=256)
    picture = models.ManyToManyField(File, blank=True, through='Match')
    labels = ArrayField(models.CharField(max_length=256))
    generated_labels = ArrayField(models.CharField(max_length=256), null=True)

    def __str__(self):
        return "{}: {}".format(self.id, self.name)


class Match(models.Model):
    file = models.ForeignKey(File, on_delete=models.CASCADE)
    dog = models.ForeignKey(Dog, on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
