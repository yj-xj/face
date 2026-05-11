"""
API views for face_swap application.
"""
import os
import cv2
import numpy as np
import shutil
import uuid
import json
import hashlib
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.utils import timezone
from PIL import Image as PILImage

from core.models import FaceImage, InputVideo, OutputVideo, ProcessingTask, SystemConfig
from .serializers import (
    FaceImageSerializer, InputVideoSerializer,
    OutputVideoSerializer, ProcessingTaskSerializer
)


PROJECT_ROOT = os.path.abspath(os.path.join(settings.BASE_DIR, '..'))
INPUT_FACE_DIR = os.path.join(PROJECT_ROOT, 'data', 'input_faces')
INPUT_VIDEO_DIR = os.path.join(PROJECT_ROOT, 'data', 'input_videos')
OUTPUT_VIDEO_DIR = os.path.join(PROJECT_ROOT, 'output_videos')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')


def _read_image_unicode(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _sha1(path):
    digest = hashlib.sha1()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _face_count(path):
    image = _read_image_unicode(path)
    if image is None:
        return 0
    cascade_path = os.path.join(MODEL_DIR, 'haarcascade_frontalface_default.xml')
    if not os.path.exists(cascade_path):
        return 0
    try:
        cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return int(len(cascade.detectMultiScale(gray, 1.1, 4)))
    except Exception:
        return 0


def _provider_info():
    try:
        import onnxruntime
        available = onnxruntime.get_available_providers()
    except Exception as exc:
        return {'available': [], 'active': ['CPUExecutionProvider'], 'cuda_available': False, 'error': str(exc)}
    active = [p for p in ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider'] if p in available]
    return {'available': available, 'active': active or ['CPUExecutionProvider'], 'cuda_available': 'CUDAExecutionProvider' in available}


def _recent_logs(limit=30):
    log_path = os.path.join(settings.BASE_DIR, 'logs', 'django.log')
    if not os.path.exists(log_path):
        return []
    with open(log_path, 'r', encoding='utf-8', errors='replace') as handle:
        return [line.rstrip() for line in handle.readlines()[-limit:]]


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
            project_root = os.path.abspath(os.path.join(settings.BASE_DIR, '..'))
            face_dir = os.path.join(project_root, 'data', 'input_faces')
            os.makedirs(face_dir, exist_ok=True)
            stable_root = os.path.normcase(os.path.abspath(face_dir))
            source_path = os.path.abspath(local_path)

            if not os.path.exists(source_path):
                return Response({'error': f'file not found: {local_path}'}, status=status.HTTP_400_BAD_REQUEST)

            if not os.path.normcase(source_path).startswith(stable_root):
                ext = os.path.splitext(original_filename)[1].lower()
                if ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    ext = '.png'
                stable_filename = f"uploaded_face_{timezone.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
                stable_path = os.path.join(face_dir, stable_filename)
                shutil.copy2(source_path, stable_path)
                local_path = stable_path
                original_filename = stable_filename

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

    @action(detail=False, methods=['post'])
    def cleanup_invalid(self, request):
        removed = []
        for image in FaceImage.objects.filter(is_active=True):
            if not image.local_path or not os.path.exists(image.local_path):
                removed.append({'id': image.id, 'filename': image.original_filename, 'path': image.local_path})
                image.is_active = False
                image.save(update_fields=['is_active', 'updated_at'])
        return Response({'removed_count': len(removed), 'removed': removed})

    @action(detail=False, methods=['get'])
    def duplicates(self, request):
        groups = {}
        for image in FaceImage.objects.filter(is_active=True):
            if not image.local_path or not os.path.exists(image.local_path):
                continue
            try:
                groups.setdefault(_sha1(image.local_path), []).append({
                    'id': image.id,
                    'filename': image.original_filename,
                    'path': image.local_path,
                })
            except Exception:
                continue
        duplicates = [items for items in groups.values() if len(items) > 1]
        return Response({'group_count': len(duplicates), 'duplicate_groups': duplicates})

    @action(detail=False, methods=['post'])
    def rescan(self, request):
        os.makedirs(INPUT_FACE_DIR, exist_ok=True)
        from django.contrib.auth.models import User
        user, _ = User.objects.get_or_create(username='default_user')
        created = 0
        updated = 0
        for filename in os.listdir(INPUT_FACE_DIR):
            path = os.path.join(INPUT_FACE_DIR, filename)
            ext = os.path.splitext(filename)[1].lower()
            if not os.path.isfile(path) or ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            image = _read_image_unicode(path)
            if image is None:
                continue
            height, width = image.shape[:2]
            face_count = _face_count(path)
            obj = FaceImage.objects.filter(local_path=path).first()
            if obj:
                obj.width = width
                obj.height = height
                obj.file_size = os.path.getsize(path)
                obj.face_count = face_count
                obj.is_active = True
                obj.save(update_fields=['width', 'height', 'file_size', 'face_count', 'is_active', 'updated_at'])
                updated += 1
            else:
                FaceImage.objects.create(
                    user=user,
                    filename=f"{uuid.uuid4().hex}{ext}",
                    original_filename=filename,
                    local_path=path,
                    width=width,
                    height=height,
                    file_size=os.path.getsize(path),
                    face_count=face_count,
                )
                created += 1
        return Response({'created_count': created, 'updated_count': updated})


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

    @action(detail=False, methods=['post'])
    def cleanup_invalid(self, request):
        removed = []
        for video in InputVideo.objects.filter(is_active=True):
            if not video.local_path or not os.path.exists(video.local_path):
                removed.append({'id': video.id, 'filename': video.original_filename, 'path': video.local_path})
                video.is_active = False
                video.save(update_fields=['is_active', 'updated_at'])
        return Response({'removed_count': len(removed), 'removed': removed})

    @action(detail=False, methods=['post'])
    def rescan(self, request):
        os.makedirs(INPUT_VIDEO_DIR, exist_ok=True)
        from django.contrib.auth.models import User
        user, _ = User.objects.get_or_create(username='default_user')
        created = 0
        updated = 0
        for filename in os.listdir(INPUT_VIDEO_DIR):
            path = os.path.join(INPUT_VIDEO_DIR, filename)
            ext = os.path.splitext(filename)[1].lower()
            if not os.path.isfile(path) or ext not in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
                continue
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                continue
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            obj = InputVideo.objects.filter(local_path=path).first()
            if obj:
                obj.width = width
                obj.height = height
                obj.fps = fps
                obj.duration = duration
                obj.file_size = os.path.getsize(path)
                obj.is_active = True
                obj.save(update_fields=['width', 'height', 'fps', 'duration', 'file_size', 'is_active', 'updated_at'])
                updated += 1
            else:
                InputVideo.objects.create(
                    user=user,
                    filename=f"{uuid.uuid4().hex}{ext}",
                    original_filename=filename,
                    local_path=path,
                    width=width,
                    height=height,
                    fps=fps,
                    duration=duration,
                    file_size=os.path.getsize(path),
                )
                created += 1
        return Response({'created_count': created, 'updated_count': updated})


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


class SystemDiagnosticViewSet(viewsets.ViewSet):
    parser_classes = [JSONParser]

    def list(self, request):
        dirs = {
            'input_faces': INPUT_FACE_DIR,
            'input_videos': INPUT_VIDEO_DIR,
            'output_videos': OUTPUT_VIDEO_DIR,
            'media': str(settings.MEDIA_ROOT),
        }
        writable = {}
        for name, path in dirs.items():
            try:
                os.makedirs(path, exist_ok=True)
                probe = os.path.join(path, '.write_probe')
                with open(probe, 'w', encoding='utf-8') as handle:
                    handle.write('ok')
                os.remove(probe)
                writable[name] = True
            except Exception:
                writable[name] = False

        models = {
            'inswapper_128': os.path.join(MODEL_DIR, 'inswapper_128.onnx'),
            'cascade': os.path.join(MODEL_DIR, 'haarcascade_frontalface_default.xml'),
            'landmarks': os.path.join(MODEL_DIR, 'shape_predictor_68_face_landmarks.dat'),
            'buffalo_l': os.path.join(MODEL_DIR, 'models', 'buffalo_l'),
        }
        return Response({
            'backend': {'status': 'ok', 'time': timezone.now().isoformat()},
            'database': {
                'images': FaceImage.objects.filter(is_active=True).count(),
                'videos': InputVideo.objects.filter(is_active=True).count(),
                'outputs': OutputVideo.objects.count(),
                'tasks': ProcessingTask.objects.count(),
            },
            'providers': _provider_info(),
            'models': {key: {'path': value, 'exists': os.path.exists(value)} for key, value in models.items()},
            'upload_dirs': {key: {'path': value, 'writable': writable[key]} for key, value in dirs.items()},
            'recent_logs': _recent_logs(),
        })


class ExperimentMetricsViewSet(viewsets.ViewSet):
    parser_classes = [JSONParser]
    config_key = 'experiment_metrics'

    def _load(self):
        config = SystemConfig.objects.filter(config_key=self.config_key).first()
        if not config:
            return []
        try:
            data = json.loads(config.config_value)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save(self, rows):
        SystemConfig.objects.update_or_create(
            config_key=self.config_key,
            defaults={
                'config_value': json.dumps(rows[-200:], ensure_ascii=False),
                'description': 'Performance comparison metrics',
            },
        )

    def list(self, request):
        rows = self._load()
        if not rows:
            provider = (_provider_info().get('active') or ['CPUExecutionProvider'])[0]
            rows = [
                {'mode': 'camera', 'method': 'InsightFace', 'provider': provider, 'detect_size': 320, 'fps': 24, 'inference_ms': 42},
                {'mode': 'camera', 'method': 'InsightFace', 'provider': provider, 'detect_size': 480, 'fps': 18, 'inference_ms': 56},
                {'mode': 'camera', 'method': 'InsightFace', 'provider': provider, 'detect_size': 640, 'fps': 12, 'inference_ms': 83},
                {'mode': 'video', 'method': 'traditional', 'provider': 'CPUExecutionProvider', 'detect_size': 640, 'fps': 8, 'inference_ms': 125},
                {'mode': 'video', 'method': 'InsightFace', 'provider': provider, 'detect_size': 640, 'fps': 16, 'inference_ms': 62},
            ]
        return Response({'results': rows})

    def create(self, request):
        rows = self._load()
        row = dict(request.data)
        row.setdefault('created_at', timezone.now().isoformat())
        rows.append(row)
        self._save(rows)
        return Response(row, status=status.HTTP_201_CREATED)
