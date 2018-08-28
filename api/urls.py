from rest_framework.routers import DefaultRouter
from django.conf.urls import url, include
from .api import FileView, DogView, MatchView

router = DefaultRouter()
router.register(
    'upload',
    viewset=FileView,
    base_name='file-upload'
)
router.register(
    'dog',
    viewset=DogView,
    base_name='dog'
)
router.register(
    'add-file-to-dog',
    viewset=MatchView,
    base_name='add-file-to-dog'
)


urlpatterns = [
    url(r'^', include(router.urls)),
]
