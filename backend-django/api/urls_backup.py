"""
URL configuration for API app.
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'images', views.FaceImageViewSet, basename='faceimage')
router.register(r'videos', views.InputVideoViewSet, basename='inputvideo')
router.register(r'outputs', views.OutputVideoViewSet, basename='outputvideo')
router.register(r'tasks', views.ProcessingTaskViewSet, basename='processingtask')

urlpatterns = [
    path('', include(router.urls)),
]
