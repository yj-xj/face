"""
API views for face_swap application.
"""
import os
import cv2
import numpy as np
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.utils import timezone
from PIL import Image as PILImage

from core.models import FaceImage, InputVideo, OutputVideo, ProcessingTask
from .serializers import (
    FaceImageSerializer, InputVideoSerializer,
    OutputVideoSerializer, ProcessingTaskSerializer
)


class FaceImageViewSet(viewsets.ModelViewSet):
    """人脸图片视图集"""
    serializer_class = FaceImageSerializer
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def get_queryset(self):
        """获取查询集"""
        queryset = FaceImage.objects.filter(is_active=True)
        user_id = self.request.query_params.get('user_id')
        if user_id:
            queryset = queryset.filter(user_id=user_id)
        return queryset

    def create(self, request, *args, **kwargs):
        """创建图片对象 - 接收本地路径"""
        local_path = request.data.get('local_path')
        original_filename = request.data.get('original_filename')

        if not local_path:
            return Response({'error': 'local_path is required'}, status=status.HTTP_400_BAD_REQUEST)

        if not original_filename:
            original_filename = os.path.basename(local_path)

        try:
            # 读取图片获取元数据
            img = PILImage.open(local_path)
            width, height = img.size
            file_size = os.path.getsize(local_path)

            # Get or create user
            from django.contrib.auth.models import User
            user, _ = User.objects.get_or_create(username='default_user')

            # 生成唯一filename
            import uuid
            ext = os.path.splitext(original_filename)[1]
            unique_filename = f"{uuid.uuid4().hex}{ext}"

            # 创建对象，只保存本地路径
            instance = FaceImage(
                user=user,
                filename=unique_filename,
                original_filename=original_filename,
                local_path=local_path,  # 保存本地路径
                file_size=file_size,
                width=width,
                height=height
            )
            instance.save()

            # 序列化返回数据
            serializer = self.get_serializer(instance)
            headers = self.get_success_headers(serializer.data)
            return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def generate_thumbnail(self, instance):
        """生成缩略图"""
        try:
            if instance.file:
                img = PILImage.open(instance.file.path)
                img.thumbnail((200, 200), PILImage.Resampling.LANCZOS)

                # 保存缩略图
                thumb_name = f'thumb_{instance.filename}'
                thumb_path = default_storage.save(f'thumbnails/{thumb_name}', ContentFile(img.tobytes()))
                instance.thumbnail = thumb_path
                instance.save()
        except Exception as e:
            print(f"生成缩略图失败: {e}")

    @action(detail=False, methods=['get'])
    def active(self, request):
        """获取所有活跃的图片"""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)


class InputVideoViewSet(viewsets.ModelViewSet):
    """输入视频视图集"""
    serializer_class = InputVideoSerializer
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def get_queryset(self):
        """获取查询集"""
        queryset = InputVideo.objects.filter(is_active=True)
        user_id = self.request.query_params.get('user_id')
        if user_id:
            queryset = queryset.filter(user_id=user_id)
        return queryset

    def create(self, request, *args, **kwargs):
        """创建视频对象 - 接收本地路径"""
        local_path = request.data.get('local_path')
        original_filename = request.data.get('original_filename')

        if not local_path:
            return Response({'error': 'local_path is required'}, status=status.HTTP_400_BAD_REQUEST)

        if not original_filename:
            original_filename = os.path.basename(local_path)

        try:
            # 读取视频获取元数据
            cap = cv2.VideoCapture(local_path)
            if not cap.isOpened():
                return Response({'error': 'Cannot open video file'}, status=status.HTTP_400_BAD_REQUEST)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()

            file_size = os.path.getsize(local_path)

            # Get or create user
            from django.contrib.auth.models import User
            user, _ = User.objects.get_or_create(username='default_user')

            # 生成唯一filename
            import uuid
            ext = os.path.splitext(original_filename)[1]
            unique_filename = f"{uuid.uuid4().hex}{ext}"

            # 创建对象，只保存本地路径
            instance = InputVideo(
                user=user,
                filename=unique_filename,
                original_filename=original_filename,
                local_path=local_path,  # 保存本地路径
                file_size=file_size,
                duration=duration,
                width=width,
                height=height,
                fps=fps
            )
            instance.save()

            # 序列化返回数据
            serializer = self.get_serializer(instance)
            headers = self.get_success_headers(serializer.data)
            return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def generate_thumbnail(self, instance, video_path=None):
        """生成视频缩略图"""
        try:
            if video_path:
                cap = cv2.VideoCapture(video_path)
            elif instance.file:
                cap = cv2.VideoCapture(instance.file.path)
            else:
                return

            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (320, 240))
                is_success, buffer = cv2.imencode(".jpg", frame)
                if is_success:
                    thumb_name = f'thumb_{instance.filename}.jpg'
                    thumb_path = default_storage.save(f'thumbnails/{thumb_name}', ContentFile(buffer.tobytes()))
                    instance.thumbnail = thumb_path
                    instance.save()

            cap.release()
        except Exception as e:
            print(f"生成缩略图失败: {e}")

    @action(detail=False, methods=['get'])
    def active(self, request):
        """获取所有活跃的视频"""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)


class OutputVideoViewSet(viewsets.ModelViewSet):
    """输出视频视图集"""
    serializer_class = OutputVideoSerializer
    parser_classes = [MultiPartParser, FormParser]

    def get_queryset(self):
        """获取查询集"""
        queryset = OutputVideo.objects.all()
        user_id = self.request.query_params.get('user_id')
        status_filter = self.request.query_params.get('status')

        if user_id:
            queryset = queryset.filter(user_id=user_id)
        if status_filter:
            queryset = queryset.filter(status=status_filter)

        return queryset

    def create(self, request, *args, **kwargs):
        """创建输出视频记录"""
        file = request.FILES.get('file')
        if not file:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # 获取关联对象
            input_video_id = request.data.get('input_video')
            face_image_id = request.data.get('face_image')

            if not face_image_id:
                return Response({'error': 'face_image_id is required'}, status=status.HTTP_400_BAD_REQUEST)

            try:
                face_image = FaceImage.objects.get(id=face_image_id)
            except FaceImage.DoesNotExist:
                return Response({'error': 'Face image not found'}, status=status.HTTP_404_NOT_FOUND)

            input_video = None
            if input_video_id:
                try:
                    input_video = InputVideo.objects.get(id=input_video_id)
                except InputVideo.DoesNotExist:
                    return Response({'error': 'Input video not found'}, status=status.HTTP_404_NOT_FOUND)

            # 获取视频元数据
            import uuid
            ext = os.path.splitext(file.name)[1]
            unique_filename = f"{uuid.uuid4().hex}{ext}"

            # 读取视频获取信息
            temp_path = default_storage.save(f'temp/{file.name}', ContentFile(file.read()))
            temp_full_path = default_storage.path(temp_path)

            cap = cv2.VideoCapture(temp_full_path)
            if not cap.isOpened():
                default_storage.delete(temp_path)
                return Response({'error': 'Cannot open video file'}, status=status.HTTP_400_BAD_REQUEST)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()

            # Get or create user
            from django.contrib.auth.models import User
            user, _ = User.objects.get_or_create(username='default_user')

            # 直接使用model创建对象
            instance = OutputVideo(
                user=user,
                face_image=face_image,
                input_video=input_video,
                filename=unique_filename,
                file_size=file.size,
                duration=duration,
                width=width,
                height=height,
                fps=fps,
                processing_method=request.data.get('processing_method', 'inswapper'),
                processing_time=request.data.get('processing_time', duration * 0.5),
                status=request.data.get('status', 'completed'),
                progress=request.data.get('progress', 100)
            )
            # 保存文件
            instance.file.save(file.name, file, save=True)

            # 删除临时文件
            default_storage.delete(temp_path)

            # 生成缩略图
            self.generate_thumbnail(instance)

            # 序列化返回数据
            serializer = self.get_serializer(instance)
            headers = self.get_success_headers(serializer.data)
            return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def generate_thumbnail(self, instance):
        """生成视频缩略图"""
        try:
            if instance.file:
                cap = cv2.VideoCapture(instance.file.path)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (320, 240))
                    is_success, buffer = cv2.imencode(".jpg", frame)
                    if is_success:
                        thumb_name = f'thumb_{instance.filename}.jpg'
                        thumb_path = default_storage.save(f'thumbnails/{thumb_name}', ContentFile(buffer.tobytes()))
                        instance.thumbnail = thumb_path
                        instance.save()
                cap.release()
        except Exception as e:
            print(f"生成缩略图失败: {e}")

    @action(detail=True, methods=['get'])
    def download(self, request, pk=None):
        """下载输出视频"""
        try:
            instance = self.get_object()
            if instance.file:
                return Response({
                    'url': instance.file.url,
                    'filename': instance.filename
                })
            return Response({'error': 'File not found'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class ProcessingTaskViewSet(viewsets.ModelViewSet):
    """处理任务视图集"""
    serializer_class = ProcessingTaskSerializer

    def get_queryset(self):
        """获取查询集"""
        queryset = ProcessingTask.objects.all()
        user_id = self.request.query_params.get('user_id')
        status_filter = self.request.query_params.get('status')
        task_type = self.request.query_params.get('task_type')

        if user_id:
            queryset = queryset.filter(user_id=user_id)
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        if task_type:
            queryset = queryset.filter(task_type=task_type)

        return queryset

    @action(detail=False, methods=['post'])
    def create_task(self, request):
        """创建处理任务"""
        try:
            face_image_id = request.data.get('face_image_id')
            input_video_id = request.data.get('input_video_id')
            processing_method = request.data.get('processing_method', 'inswapper')
            processing_params = request.data.get('processing_params', {})

            if not face_image_id:
                return Response({'error': 'face_image_id is required'}, status=status.HTTP_400_BAD_REQUEST)

            # 获取对象
            try:
                face_image = FaceImage.objects.get(id=face_image_id)
            except FaceImage.DoesNotExist:
                return Response({'error': 'Face image not found'}, status=status.HTTP_404_NOT_FOUND)

            input_video = None
            if input_video_id:
                try:
                    input_video = InputVideo.objects.get(id=input_video_id)
                except InputVideo.DoesNotExist:
                    return Response({'error': 'Input video not found'}, status=status.HTTP_404_NOT_FOUND)

            # 创建任务
            task = ProcessingTask.objects.create(
                user=request.user,
                task_type='video' if input_video else 'camera',
                face_image=face_image,
                input_video=input_video,
                processing_params={
                    **processing_params,
                    'processing_method': processing_method
                },
                status='pending'
            )

            serializer = self.get_serializer(task)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        """取消任务"""
        try:
            task = self.get_object()
            if task.status in ['pending', 'processing']:
                task.status = 'failed'
                task.error_message = 'Task cancelled by user'
                task.completed_at = timezone.now()
                task.save()
                serializer = self.get_serializer(task)
                return Response(serializer.data)
            return Response({'error': 'Task cannot be cancelled'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
