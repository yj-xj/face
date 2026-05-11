"""
URL configuration for API app.
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework.authtoken import views as auth_views
from . import views

router = DefaultRouter()
router.register(r'images', views.FaceImageViewSet, basename='faceimage')
router.register(r'videos', views.InputVideoViewSet, basename='inputvideo')
router.register(r'outputs', views.OutputVideoViewSet, basename='outputvideo')
router.register(r'tasks', views.ProcessingTaskViewSet, basename='processingtask')
router.register(r'diagnostics', views.SystemDiagnosticViewSet, basename='diagnostics')
router.register(r'experiments', views.ExperimentMetricsViewSet, basename='experiments')

urlpatterns = [
    path('', include(router.urls)),
    path('upload/image/', views.FaceImageViewSet.as_view({'post': 'create'})),
    path('upload/video/', views.InputVideoViewSet.as_view({'post': 'create'})),
]
