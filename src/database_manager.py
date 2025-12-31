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

    def __init__(self, base_url="http://localhost:5000/api"):
        super().__init__()
        self.base_url = base_url
        self.user_id = 1  # 默认用户ID

    def check_server(self):
        """检查服务器连接"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
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

    def run(self):
        try:
            self.progress.emit(0, "准备上传...")

            if not os.path.exists(self.file_path):
                self.error.emit(f"文件不存在: {self.file_path}")
                return

            self.progress.emit(20, "读取文件...")
            with open(self.file_path, 'rb') as f:
                files = {'file': (os.path.basename(self.file_path), f, 'image/jpeg')}
                data = {'user_id': self.user_id}

                self.progress.emit(50, "上传中...")
                response = requests.post(
                    f"{self.base_url}/upload/image",
                    files=files,
                    data=data,
                    timeout=300
                )

                self.progress.emit(90, "处理响应...")

                if response.status_code == 200:
                    result = response.json()
                    self.finished.emit(result)
                    self.progress.emit(100, "上传成功")
                else:
                    self.error.emit(f"上传失败: {response.status_code}")

        except Exception as e:
            self.error.emit(f"上传错误: {str(e)}")


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

    def run(self):
        try:
            self.progress.emit(0, "准备上传...")

            if not os.path.exists(self.file_path):
                self.error.emit(f"文件不存在: {self.file_path}")
                return

            self.progress.emit(20, "读取文件...")
            file_size = os.path.getsize(self.file_path)

            with open(self.file_path, 'rb') as f:
                files = {'file': (os.path.basename(self.file_path), f, 'video/mp4')}
                data = {'user_id': self.user_id}

                self.progress.emit(50, "上传中...")
                response = requests.post(
                    f"{self.base_url}/upload/video",
                    files=files,
                    data=data,
                    timeout=600
                )

                self.progress.emit(90, "处理响应...")

                if response.status_code == 200:
                    result = response.json()
                    self.finished.emit(result)
                    self.progress.emit(100, "上传成功")
                else:
                    self.error.emit(f"上传失败: {response.status_code}")

        except Exception as e:
            self.error.emit(f"上传错误: {str(e)}")


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
                f"{self.base_url}/images",
                params={'user_id': self.user_id, 'per_page': 1000},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                self.finished.emit(result.get('images', []))
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
                f"{self.base_url}/videos",
                params={'user_id': self.user_id, 'per_page': 1000},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                self.finished.emit(result.get('videos', []))
            else:
                self.error.emit(f"加载失败: {response.status_code}")

        except Exception as e:
            self.error.emit(f"加载错误: {str(e)}")
