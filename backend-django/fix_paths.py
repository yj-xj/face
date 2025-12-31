#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修正数据库中的路径格式
"""
import os
import sys
import django

# 设置Django环境
sys.path.insert(0, 'E:/face/backend-django')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_swap.settings')
django.setup()

from core.models import FaceImage, InputVideo

print("="*80)
print("修正数据库路径格式")
print("="*80)

# 修正图片路径
print("\n[1] 修正图片路径:")
print("-"*80)
images = FaceImage.objects.all()
print(f"总共 {images.count()} 个图片")

updated = 0
for img in images:
    if img.local_path:
        # 规范化路径
        original_path = img.local_path
        normalized_path = os.path.normpath(img.local_path)
        # 转换为绝对路径
        absolute_path = os.path.abspath(normalized_path)

        if original_path != absolute_path:
            # 检查文件是否存在
            if os.path.exists(absolute_path):
                img.local_path = absolute_path
                img.save()
                print(f"[更新] ID:{img.id} - {img.original_filename}")
                print(f"  旧路径: {original_path}")
                print(f"  新路径: {absolute_path}")
                print(f"  文件存在: 是")
                updated += 1
            else:
                print(f"[跳过] ID:{img.id} - {img.original_filename}")
                print(f"  路径: {absolute_path}")
                print(f"  文件不存在")

print(f"\n更新了 {updated} 个图片路径")

# 修正视频路径
print("\n[2] 修正视频路径:")
print("-"*80)
videos = InputVideo.objects.all()
print(f"总共 {videos.count()} 个视频")

updated = 0
for video in videos:
    if video.local_path:
        # 规范化路径
        original_path = video.local_path
        normalized_path = os.path.normpath(video.local_path)
        # 转换为绝对路径
        absolute_path = os.path.abspath(normalized_path)

        if original_path != absolute_path:
            # 检查文件是否存在
            if os.path.exists(absolute_path):
                video.local_path = absolute_path
                video.save()
                print(f"[更新] ID:{video.id} - {video.original_filename}")
                print(f"  旧路径: {original_path}")
                print(f"  新路径: {absolute_path}")
                print(f"  文件存在: 是")
                updated += 1
            else:
                print(f"[跳过] ID:{video.id} - {video.original_filename}")
                print(f"  路径: {absolute_path}")
                print(f"  文件不存在")

print(f"\n更新了 {updated} 个视频路径")

# 统计有效路径
print("\n" + "="*80)
print("[3] 统计有效路径:")
print("-"*80)

images_with_path = FaceImage.objects.exclude(local_path__isnull=True).exclude(local_path__exact='')
print(f"有local_path的图片: {images_with_path.count()}")

valid_images = 0
for img in images_with_path:
    if os.path.exists(img.local_path):
        valid_images += 1

print(f"路径有效的图片: {valid_images}")

videos_with_path = InputVideo.objects.exclude(local_path__isnull=True).exclude(local_path__exact='')
print(f"\n有local_path的视频: {videos_with_path.count()}")

valid_videos = 0
for video in videos_with_path:
    if os.path.exists(video.local_path):
        valid_videos += 1

print(f"路径有效的视频: {valid_videos}")

print("\n" + "="*80)
print("修正完成")
print("="*80)
