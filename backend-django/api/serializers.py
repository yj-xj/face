"""
Serializers for face_swap API.
"""
from rest_framework import serializers
from core.models import FaceImage, InputVideo, OutputVideo, ProcessingTask, SystemConfig


class FaceImageSerializer(serializers.ModelSerializer):
    """人脸图片序列化器"""
    file_url = serializers.SerializerMethodField()
    thumbnail_url = serializers.SerializerMethodField()
    user_username = serializers.CharField(source='user.username', read_only=True)

    class Meta:
        model = FaceImage
        fields = [
            'id', 'user', 'user_username', 'filename', 'original_filename',
            'file', 'thumbnail', 'file_url', 'thumbnail_url',
            'width', 'height', 'file_size', 'face_count', 'is_active',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['user', 'file_size', 'width', 'height', 'created_at', 'updated_at']

    def get_file_url(self, obj):
        if obj.file:
            return obj.file.url
        return None

    def get_thumbnail_url(self, obj):
        if obj.thumbnail:
            return obj.thumbnail.url
        return None


class InputVideoSerializer(serializers.ModelSerializer):
    """输入视频序列化器"""
    file_url = serializers.SerializerMethodField()
    thumbnail_url = serializers.SerializerMethodField()
    user_username = serializers.CharField(source='user.username', read_only=True)

    class Meta:
        model = InputVideo
        fields = [
            'id', 'user', 'user_username', 'filename', 'original_filename',
            'file', 'thumbnail', 'file_url', 'thumbnail_url',
            'duration', 'width', 'height', 'fps', 'codec', 'file_size', 'is_active',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['user', 'file_size', 'duration', 'width', 'height', 'fps', 'created_at', 'updated_at']

    def get_file_url(self, obj):
        if obj.file:
            return obj.file.url
        return None

    def get_thumbnail_url(self, obj):
        if obj.thumbnail:
            return obj.thumbnail.url
        return None


class OutputVideoSerializer(serializers.ModelSerializer):
    """输出视频序列化器"""
    file_url = serializers.SerializerMethodField()
    thumbnail_url = serializers.SerializerMethodField()
    face_image_data = FaceImageSerializer(source='face_image', read_only=True)
    input_video_filename = serializers.CharField(source='input_video.filename', read_only=True)

    class Meta:
        model = OutputVideo
        fields = [
            'id', 'user', 'filename', 'file', 'file_url', 'thumbnail_url',
            'face_image', 'face_image_data', 'input_video', 'input_video_filename',
            'duration', 'width', 'height', 'fps', 'file_size',
            'processing_method', 'processing_time', 'status', 'progress', 'error_message',
            'created_at', 'completed_at'
        ]
        read_only_fields = ['user', 'file_size', 'created_at', 'completed_at']

    def get_file_url(self, obj):
        if obj.file:
            return obj.file.url
        return None

    def get_thumbnail_url(self, obj):
        if obj.thumbnail:
            return obj.thumbnail.url
        return None


class ProcessingTaskSerializer(serializers.ModelSerializer):
    """处理任务序列化器"""
    face_image_data = FaceImageSerializer(source='face_image', read_only=True)
    input_video_data = InputVideoSerializer(source='input_video', read_only=True)
    output_video_data = OutputVideoSerializer(source='output_video', read_only=True)

    class Meta:
        model = ProcessingTask
        fields = [
            'id', 'user', 'task_type', 'input_video', 'face_image', 'output_video',
            'face_image_data', 'input_video_data', 'output_video_data',
            'status', 'progress', 'error_message', 'processing_params',
            'created_at', 'updated_at', 'completed_at'
        ]
        read_only_fields = ['user', 'created_at', 'updated_at', 'completed_at']


class SystemConfigSerializer(serializers.ModelSerializer):
    """系统配置序列化器"""
    class Meta:
        model = SystemConfig
        fields = ['id', 'config_key', 'config_value', 'description', 'created_at', 'updated_at']
        read_only_fields = ['created_at', 'updated_at']
