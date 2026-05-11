#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据库管理模块 - 用于与后端API交互
"""
import os
import sys
import requests
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication, QMessageBox

class DatabaseManager(QObject):
    """数据库管理器"""
    # 信号
    image_uploaded = pyqtSignal(dict)
    video_uploaded = pyqtSignal(dict)
    images_loaded = pyqtSignal(list)
    videos_loaded = pyqtSignal(list)
    upload_progress = pyqtSignal(int, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, base_url="http://localhost:8008/api"):
        super().__init__()
        self.base_url = base_url
        self.user_id = 1  # 默认用户ID

    def check_server(self):
        """检查服务器连接"""
        try:
            # 尝试访问API根端点
            response = requests.get(f"{self.base_url}/../", timeout=5)
            return response.status_code in [200, 404]  # 404也可以，说明服务器在运行
        except Exception as e:
            print(f"服务器连接失败: {e}")
            return False

    def upload_image(self, file_path):
        """上传图片"""
        uploader = ImageUploadThread(self.base_url, file_path, self.user_id)
        uploader.finished.connect(self.image_uploaded.emit)
        uploader.error.connect(self.error_occurred.emit)
        uploader.progress.connect(self.upload_progress.emit)
        uploader.start()
        return uploader

    def upload_video(self, file_path):
        """上传视频"""
        uploader = VideoUploadThread(self.base_url, file_path, self.user_id)
        uploader.finished.connect(self.video_uploaded.emit)
        uploader.error.connect(self.error_occurred.emit)
        uploader.progress.connect(self.upload_progress.emit)
        uploader.start()
        return uploader

    def load_images(self):
        """加载所有图片"""
        loader = ImageLoadThread(self.base_url, self.user_id)
        loader.finished.connect(self.images_loaded.emit)
        loader.error.connect(self.error_occurred.emit)
        loader.start()
        return loader

    def load_videos(self):
        """加载所有视频"""
        loader = VideoLoadThread(self.base_url, self.user_id)
        loader.finished.connect(self.videos_loaded.emit)
        loader.error.connect(self.error_occurred.emit)
        loader.start()
        return loader

    def get_image_url(self, image_id):
        """获取图片URL"""
        return f"{self.base_url}/images/{image_id}"

    def get_video_url(self, video_id):
        """获取视频URL"""
        return f"{self.base_url}/videos/{video_id}"

    def get_thumbnail_url(self, file_id):
        """获取缩略图URL"""
        return f"{self.base_url}/thumbnails/{file_id}"

    def delete_image(self, image_id):
        """删除图片"""
        try:
            response = requests.delete(f"{self.base_url}/images/{image_id}")
            return response.status_code == 200
        except Exception as e:
            print(f"删除图片失败: {e}")
            return False

    def delete_video(self, video_id):
        """删除视频"""
        try:
            response = requests.delete(f"{self.base_url}/videos/{video_id}")
            return response.status_code == 200
        except Exception as e:
            print(f"删除视频失败: {e}")
            return False

    def save_output_video(self, output_path, input_video_id, face_image_id, processing_method='inswapper'):
        """保存输出视频到数据库"""
        import cv2
        import uuid

        try:
            if not os.path.exists(output_path):
                print(f"输出视频文件不存在: {output_path}")
                return None

            # 获取视频信息
            cap = cv2.VideoCapture(output_path)
            if not cap.isOpened():
                print(f"无法打开视频文件: {output_path}")
                return None

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            file_size = os.path.getsize(output_path)
            cap.release()

            # 生成唯一filename
            filename = os.path.basename(output_path)

            # 准备上传数据
            data = {
                'input_video': input_video_id,
                'face_image': face_image_id,
                'filename': filename,
                'file_size': file_size,
                'duration': duration,
                'width': width,
                'height': height,
                'fps': fps,
                'processing_method': processing_method,
                'processing_time': duration * 0.5,  # 估计处理时间
                'status': 'completed',
                'progress': 100
            }

            # 上传文件
            with open(output_path, 'rb') as f:
                files = {'file': (filename, f, 'video/mp4')}
                response = requests.post(
                    f"{self.base_url}/../outputs/",
                    files=files,
                    data=data,
                    timeout=600
                )

            if response.status_code in [200, 201]:
                result = response.json()
                print(f"输出视频已保存到数据库: {filename}")
                return result
            else:
                print(f"保存输出视频失败: {response.status_code} - {response.text[:200]}")
                return None

        except Exception as e:
            print(f"保存输出视频时出错: {e}")
            import traceback
            traceback.print_exc()
            return None


class ImageUploadThread(QThread):
    """图片上传线程"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, base_url, file_path, user_id):
        super().__init__()
        self.base_url = base_url
        self.file_path = file_path
        self.user_id = user_id
        
        # Detect MIME type
        self.mime_type = 'image/jpeg'
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.png':
            self.mime_type = 'image/png'
        elif ext in ['.jpg', '.jpeg']:
            self.mime_type = 'image/jpeg'
        elif ext == '.bmp':
            self.mime_type = 'image/bmp'

    def run(self):
        try:
            self.progress.emit(0, "准备保存...")

            if not os.path.exists(self.file_path):
                self.error.emit(f"文件不存在: {self.file_path}")
                return

            self.progress.emit(20, "读取文件信息...")

            # Get file size
            file_size = os.path.getsize(self.file_path)
            original_filename = os.path.basename(self.file_path)

            # 传递本地路径而不是上传文件
            data = {
                'local_path': self.file_path,
                'original_filename': original_filename
            }

            self.progress.emit(50, "保存到数据库...")
            print(f"Saving image: {original_filename} ({file_size} bytes) to {self.base_url}/images/")

            response = requests.post(
                f"{self.base_url}/images/",
                json=data,
                timeout=300
            )

            self.progress.emit(90, "处理响应...")
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.text[:500]}")

            if response.status_code in [200, 201]:
                result = response.json()
                self.finished.emit(result)
                self.progress.emit(100, "保存成功")
                print("Save successful!")
            else:
                error_msg = f"保存失败: {response.status_code} - {response.text[:200]}"
                self.error.emit(error_msg)
                print(f"Save failed: {error_msg}")

        except requests.exceptions.Timeout:
            error_msg = "保存超时，请检查网络连接"
            self.error.emit(error_msg)
            print(f"Save timeout: {error_msg}")
        except requests.exceptions.ConnectionError:
            error_msg = "无法连接到服务器，请确保后端已启动"
            self.error.emit(error_msg)
            print(f"Connection error: {error_msg}")
        except Exception as e:
            error_msg = f"保存错误: {str(e)}"
            self.error.emit(error_msg)
            print(f"Save error: {error_msg}")
            import traceback
            traceback.print_exc()


class VideoUploadThread(QThread):
    """视频上传线程"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, base_url, file_path, user_id):
        super().__init__()
        self.base_url = base_url
        self.file_path = file_path
        self.user_id = user_id
        
        # Detect MIME type
        self.mime_type = 'video/mp4'
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.mp4':
            self.mime_type = 'video/mp4'
        elif ext == '.avi':
            self.mime_type = 'video/x-msvideo'
        elif ext == '.mov':
            self.mime_type = 'video/quicktime'
        elif ext == '.mkv':
            self.mime_type = 'video/x-matroska'
        elif ext == '.flv':
            self.mime_type = 'video/x-flv'
        elif ext == '.wmv':
            self.mime_type = 'video/x-ms-wmv'

    def run(self):
        try:
            self.progress.emit(0, "准备保存...")

            if not os.path.exists(self.file_path):
                self.error.emit(f"文件不存在: {self.file_path}")
                return

            self.progress.emit(20, "读取文件信息...")
            file_size = os.path.getsize(self.file_path)
            original_filename = os.path.basename(self.file_path)

            # 传递本地路径而不是上传文件
            data = {
                'local_path': self.file_path,
                'original_filename': original_filename
            }

            self.progress.emit(50, "保存到数据库...")
            print(f"Saving video: {original_filename} ({file_size} bytes) to {self.base_url}/videos/")

            response = requests.post(
                f"{self.base_url}/videos/",
                json=data,
                timeout=600
            )

            self.progress.emit(90, "处理响应...")
            print(f"Response status: {response.status_code}")

            if response.status_code in [200, 201]:
                result = response.json()
                self.finished.emit(result)
                self.progress.emit(100, "保存成功")
                print("Video save successful!")
            else:
                error_msg = f"保存失败: {response.status_code} - {response.text[:200]}"
                self.error.emit(error_msg)
                print(f"Save failed: {error_msg}")

        except requests.exceptions.Timeout:
            error_msg = "保存超时，请检查网络连接"
            self.error.emit(error_msg)
            print(f"Save timeout: {error_msg}")
        except requests.exceptions.ConnectionError:
            error_msg = "无法连接到服务器，请确保后端已启动"
            self.error.emit(error_msg)
            print(f"Connection error: {error_msg}")
        except Exception as e:
            error_msg = f"保存错误: {str(e)}"
            self.error.emit(error_msg)
            print(f"Save error: {error_msg}")
            import traceback
            traceback.print_exc()


class ImageLoadThread(QThread):
    """图片加载线程"""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, base_url, user_id):
        super().__init__()
        self.base_url = base_url
        self.user_id = user_id

    def run(self):
        try:
            response = requests.get(
                f"{self.base_url}/images/",
                params={'user_id': self.user_id, 'per_page': 1000},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                # API返回 {'count': ..., 'results': [...]}
                self.finished.emit(result.get('results', []))
            else:
                self.error.emit(f"加载失败: {response.status_code}")

        except Exception as e:
            self.error.emit(f"加载错误: {str(e)}")


class VideoLoadThread(QThread):
    """视频加载线程"""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, base_url, user_id):
        super().__init__()
        self.base_url = base_url
        self.user_id = user_id

    def run(self):
        try:
            response = requests.get(
                f"{self.base_url}/videos/",
                params={'user_id': self.user_id, 'per_page': 1000},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                # API返回 {'count': ..., 'results': [...]}
                self.finished.emit(result.get('results', []))
            else:
                self.error.emit(f"加载失败: {response.status_code}")

        except Exception as e:
            self.error.emit(f"加载错误: {str(e)}")
