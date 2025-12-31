"""
Admin configuration for core models.
"""
from django.contrib import admin
from .models import FaceImage, InputVideo, OutputVideo, ProcessingTask, SystemConfig


@admin.register(FaceImage)
class FaceImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'original_filename', 'width', 'height', 'file_size', 'is_active', 'created_at']
    list_filter = ['is_active', 'created_at']
    search_fields = ['original_filename', 'filename']
    readonly_fields = ['created_at', 'updated_at', 'file_size', 'width', 'height']


@admin.register(InputVideo)
class InputVideoAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'original_filename', 'duration', 'width', 'height', 'fps', 'is_active', 'created_at']
    list_filter = ['is_active', 'created_at']
    search_fields = ['original_filename', 'filename']
    readonly_fields = ['created_at', 'updated_at', 'file_size', 'duration', 'width', 'height', 'fps']


@admin.register(OutputVideo)
class OutputVideoAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'filename', 'processing_method', 'status', 'progress', 'created_at']
    list_filter = ['status', 'processing_method', 'created_at']
    search_fields = ['filename']
    readonly_fields = ['created_at', 'completed_at']


@admin.register(ProcessingTask)
class ProcessingTaskAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'task_type', 'status', 'progress', 'created_at']
    list_filter = ['status', 'task_type', 'created_at']
    readonly_fields = ['created_at', 'updated_at', 'completed_at']


@admin.register(SystemConfig)
class SystemConfigAdmin(admin.ModelAdmin):
    list_display = ['config_key', 'config_value', 'description', 'updated_at']
    search_fields = ['config_key', 'description']
