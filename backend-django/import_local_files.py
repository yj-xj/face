#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将本地文件夹中的文件导入到数据库
"""
import os
import sys
import django

# 设置Django环境
sys.path.insert(0, 'E:/face/backend-django')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_swap.settings')
django.setup()

from core.models import FaceImage, InputVideo, OutputVideo
from django.contrib.auth.models import User
import uuid
import cv2
from PIL import Image as PILImage


def import_face_images(user, input_faces_dir='e:/face/data/input_faces'):
    """导入人脸图片"""
    print(f"\n{'='*60}")
    print(f"开始导入人脸图片: {input_faces_dir}")
    print(f"{'='*60}")

    if not os.path.exists(input_faces_dir):
        print(f"目录不存在: {input_faces_dir}")
        return

    files = [f for f in os.listdir(input_faces_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]

    print(f"找到 {len(files)} 个图片文件")

    imported = 0
    skipped = 0
    updated = 0

    for filename in files:
        file_path = os.path.join(input_faces_dir, filename)

        try:
            # 读取图片获取元数据
            img = PILImage.open(file_path)
            width, height = img.size
            file_size = os.path.getsize(file_path)

            # 检查是否已经存在（通过local_path查找）
            existing = FaceImage.objects.filter(local_path=file_path).first()

            if existing:
                print(f"[跳过] {filename} - 已存在 (ID: {existing.id})")
                skipped += 1
                continue

            # 生成唯一filename
            ext = os.path.splitext(filename)[1]
            unique_filename = f"{uuid.uuid4().hex}{ext}"

            # 创建对象
            face_image = FaceImage(
                user=user,
                filename=unique_filename,
                original_filename=filename,
                local_path=file_path,  # 保存本地路径
                file_size=file_size,
                width=width,
                height=height,
                face_count=1  # 默认值
            )
            face_image.save()

            print(f"[导入] {filename} -> ID: {face_image.id}, 路径: {file_path}")
            imported += 1

        except Exception as e:
            print(f"[错误] {filename}: {e}")

    print(f"\n导入完成: {imported} 个新文件, {skipped} 个跳过, {updated} 个更新")


def import_input_videos(user, input_videos_dir='e:/face/data/input_videos'):
    """导入输入视频"""
    print(f"\n{'='*60}")
    print(f"开始导入输入视频: {input_videos_dir}")
    print(f"{'='*60}")

    if not os.path.exists(input_videos_dir):
        print(f"目录不存在: {input_videos_dir}")
        return

    files = [f for f in os.listdir(input_videos_dir)
             if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'))]

    print(f"找到 {len(files)} 个视频文件")

    imported = 0
    skipped = 0
    updated = 0

    for filename in files:
        file_path = os.path.join(input_videos_dir, filename)

        try:
            # 读取视频获取元数据
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(f"[错误] {filename}: 无法打开视频文件")
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()

            file_size = os.path.getsize(file_path)

            # 检查是否已经存在
            existing = InputVideo.objects.filter(local_path=file_path).first()

            if existing:
                print(f"[跳过] {filename} - 已存在 (ID: {existing.id})")
                skipped += 1
                continue

            # 生成唯一filename
            ext = os.path.splitext(filename)[1]
            unique_filename = f"{uuid.uuid4().hex}{ext}"

            # 创建对象
            input_video = InputVideo(
                user=user,
                filename=unique_filename,
                original_filename=filename,
                local_path=file_path,  # 保存本地路径
                file_size=file_size,
                duration=duration,
                width=width,
                height=height,
                fps=fps
            )
            input_video.save()

            print(f"[导入] {filename} -> ID: {input_video.id}, 路径: {file_path}")
            imported += 1

        except Exception as e:
            print(f"[错误] {filename}: {e}")

    print(f"\n导入完成: {imported} 个新文件, {skipped} 个跳过, {updated} 个更新")


def import_output_videos(user, output_videos_dir='e:/face/output_videos'):
    """导出输出视频"""
    print(f"\n{'='*60}")
    print(f"开始导入输出视频: {output_videos_dir}")
    print(f"{'='*60}")

    if not os.path.exists(output_videos_dir):
        print(f"目录不存在: {output_videos_dir}")
        return

    files = [f for f in os.listdir(output_videos_dir)
             if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    print(f"找到 {len(files)} 个视频文件")

    imported = 0
    skipped = 0

    for filename in files:
        file_path = os.path.join(output_videos_dir, filename)

        try:
            # 读取视频获取元数据
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(f"[错误] {filename}: 无法打开视频文件")
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()

            file_size = os.path.getsize(file_path)

            # 检查是否已经存在
            existing = OutputVideo.objects.filter(filename=filename).first()

            if existing:
                print(f"[跳过] {filename} - 已存在 (ID: {existing.id})")
                skipped += 1
                continue

            # 生成唯一filename
            ext = os.path.splitext(filename)[1]
            unique_filename = f"{uuid.uuid4().hex}{ext}"

            # 注意：OutputVideo需要关联face_image，这里设为null
            output_video = OutputVideo(
                user=user,
                filename=unique_filename,
                local_path=file_path,
                file_size=file_size,
                duration=duration,
                width=width,
                height=height,
                fps=fps,
                processing_method='inswapper',
                processing_time=duration * 0.5,
                status='completed',
                progress=100
            )

            # 读取文件内容
            with open(file_path, 'rb') as f:
                from django.core.files.base import ContentFile
                output_video.file.save(filename, ContentFile(f.read()), save=True)

            print(f"[导入] {filename} -> ID: {output_video.id}, 路径: {file_path}")
            imported += 1

        except Exception as e:
            print(f"[错误] {filename}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n导入完成: {imported} 个新文件, {skipped} 个跳过")


if __name__ == '__main__':
    # 获取或创建默认用户
    user, _ = User.objects.get_or_create(username='default_user')

    print("="*60)
    print("本地文件导入工具")
    print("="*60)

    # 导入人脸图片
    import_face_images(user, 'e:/face/data/input_faces')

    # 导入输入视频
    import_input_videos(user, 'e:/face/data/input_videos')

    # 导入输出视频（可选）
    # import_output_videos(user, 'e:/face/output_videos')

    print("\n" + "="*60)
    print("全部导入完成！")
    print("="*60)
