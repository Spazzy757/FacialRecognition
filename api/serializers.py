from rest_framework import serializers
from .models import File, Dog, Match


class FileSerializer(serializers.ModelSerializer):
    class Meta:
        model = File
        fields = ('id', 'file', 'remark', 'timestamp')


class DogSerializer(serializers.ModelSerializer):

    class Meta:
        model = Dog
        fields = ('id', 'name', 'labels', 'generated_labels')
        read_only = ('generated_labels',)


class MatchSerializer(serializers.ModelSerializer):

    class Meta:
        model = Match
        fields = ('id', 'file', 'dog')
