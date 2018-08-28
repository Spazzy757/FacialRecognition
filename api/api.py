from .serializers import FileSerializer, DogSerializer, MatchSerializer
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.viewsets import ModelViewSet
from api.models import File, Dog, Match


class FileView(ModelViewSet):
    queryset = File.objects.all()
    serializer_class = FileSerializer
    parser_classes = (MultiPartParser, FormParser)


class DogView(ModelViewSet):
    queryset = Dog.objects.all()
    serializer_class = DogSerializer


class MatchView(ModelViewSet):
    queryset = Match.objects.all()
    serializer_class = MatchSerializer
