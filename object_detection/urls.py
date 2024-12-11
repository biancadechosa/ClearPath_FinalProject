from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Main page
    path('detect', views.detect_objects, name='detect_objects'),  # Add the detect route
]
