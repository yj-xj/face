"""
Core models for face_swap application.
"""
import os
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator
from django.utils import timezone


def get_upload_path(instance, filename):
    """Generate upload path for files."""
    now = timezone.now()
    return os.path.join(
        now.strftime('%Y'),
        now.strftime('%m'),
        now.strftime('%d'),
        filename
    )


class FaceImage(models.Model):
    """人脸图片模型"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='face_images')
    filename = models.CharField(max_length=255, unique=True)
    original_filename = models.CharField(max_length=255)
    file = models.ImageField(upload_to=get_upload_path)
    thumbnail = models.ImageField(upload_to=get_upload_path, blank=True, null=True)

    # Image metadata
    width = models.IntegerField(blank=True, null=True)
    height = models.IntegerField(blank=True, null=True)
    file_size = models.BigInteger(blank=True, null=True)
    face_count = models.IntegerField(default=0)

    # Status
    is_active = models.BooleanField(default=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'face_images'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'is_active']),
            models.Index(fields=['-created_at']),
        ]

    def __str__(self):
        return self.original_filename

    def delete(self, *args, **kwargs):
        """删除文件"""
        # Delete actual files
        if self.file and os.path.exists(self.file.path):
            os.remove(self.file.path)
        if self.thumbnail and os.path.exists(self.thumbnail.path):
            os.remove(self.thumbnail.path)
        super().delete(*args, **kwargs)


class InputVideo(models.Model):
    """输入视频模型"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='input_videos')
    filename = models.CharField(max_length=255, unique=True)
    original_filename = models.CharField(max_length=255)
    file = models.FileField(upload_to=get_upload_path)
    thumbnail = models.ImageField(upload_to=get_upload_path, blank=True, null=True)

    # Video metadata
    duration = models.FloatField(blank=True, null=True, help_text='Duration in seconds')
    width = models.IntegerField(blank=True, null=True)
    height = models.IntegerField(blank=True, null=True)
    fps = models.FloatField(blank=True, null=True)
    codec = models.CharField(max_length=50, blank=True)
    file_size = models.BigInteger(blank=True, null=True)

    # Status
    is_active = models.BooleanField(default=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'input_videos'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'is_active']),
            models.Index(fields=['-created_at']),
        ]

    def __str__(self):
        return self.original_filename

    def delete(self, *args, **kwargs):
        """删除文件"""
        if self.file and os.path.exists(self.file.path):
            os.remove(self.file.path)
        if self.thumbnail and os.path.exists(self.thumbnail.path):
            os.remove(self.thumbnail.path)
        super().delete(*args, **kwargs)


class OutputVideo(models.Model):
    """输出视频模型"""
    PROCESSING_METHOD_CHOICES = [
        ('traditional', 'Traditional'),
        ('inswapper', 'Inswapper'),
    ]
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='output_videos')
    input_video = models.ForeignKey(InputVideo, on_delete=models.SET_NULL, null=True, blank=True, related_name='outputs')
    face_image = models.ForeignKey(FaceImage, on_delete=models.CASCADE, related_name='outputs')

    filename = models.CharField(max_length=255, unique=True)
    file = models.FileField(upload_to=get_upload_path)
    thumbnail = models.ImageField(upload_to=get_upload_path, blank=True, null=True)

    # Video metadata
    duration = models.FloatField(blank=True, null=True)
    width = models.IntegerField(blank=True, null=True)
    height = models.IntegerField(blank=True, null=True)
    fps = models.FloatField(blank=True, null=True)
    file_size = models.BigInteger(blank=True, null=True)

    # Processing info
    processing_method = models.CharField(max_length=20, choices=PROCESSING_METHOD_CHOICES, default='inswapper')
    processing_time = models.FloatField(blank=True, null=True, help_text='Processing time in seconds')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    error_message = models.TextField(blank=True)

    # Progress
    progress = models.IntegerField(default=0, help_text='Progress 0-100')

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        db_table = 'output_videos'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'status']),
            models.Index(fields=['-created_at']),
        ]

    def __str__(self):
        return self.filename


class ProcessingTask(models.Model):
    """处理任务模型"""
    TASK_TYPE_CHOICES = [
        ('video', 'Video Processing'),
        ('camera', 'Camera Processing'),
    ]
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='tasks')
    task_type = models.CharField(max_length=20, choices=TASK_TYPE_CHOICES)

    input_video = models.ForeignKey(InputVideo, on_delete=models.SET_NULL, null=True, blank=True, related_name='tasks')
    face_image = models.ForeignKey(FaceImage, on_delete=models.CASCADE, related_name='tasks')
    output_video = models.OneToOneField(OutputVideo, on_delete=models.SET_NULL, null=True, blank=True, related_name='task')

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    progress = models.IntegerField(default=0, help_text='Progress 0-100')
    error_message = models.TextField(blank=True)

    # Processing parameters (JSON)
    processing_params = models.JSONField(default=dict, blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        db_table = 'processing_tasks'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'status']),
            models.Index(fields=['-created_at']),
        ]

    def __str__(self):
        return f"{self.get_task_type_display()} - {self.id}"


class SystemConfig(models.Model):
    """系统配置模型"""
    config_key = models.CharField(max_length=100, unique=True)
    config_value = models.TextField()
    description = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'system_config'
        indexes = [
            models.Index(fields=['config_key']),
        ]

    def __str__(self):
        return self.config_key
