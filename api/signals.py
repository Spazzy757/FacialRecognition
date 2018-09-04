from django.db.models.signals import pre_save
from engine.recognition import BreedClassifier
from django.dispatch import receiver
from api.models import Match, Dog, Label
import json


@receiver(pre_save, sender=Match)
def classify(instance, **kwargs):
    classifier = BreedClassifier()
    result = classifier.classify('/code/media/images/{}'.format(
        instance.file.file.name)
    )
    import ipdb;
    ipdb.set_trace()
    dog = Dog.objects.get(id=instance.dog.id)
    for prediction in result.get('results'):
        Label.objects.get_or_create(dog_id=dog.id, **prediction)

