#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整测试前端加载流程
"""
import requests
import os

API_BASE = "http://localhost:8000/api"

print("="*80)
print("完整测试前端加载流程")
print("="*80)

# 1. 测试获取图片
print("\n[1] 测试获取图片:")
print("-"*80)
response = requests.get(f"{API_BASE}/images/", params={'user_id': 1, 'per_page': 20})
print(f"状态码: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    images = data.get('results', [])
    print(f"获取到 {len(images)} 个图片")

    # 模拟前端 onImagesLoaded 处理
    valid_images = []
    for image in images:
        local_path = image.get('local_path')
        if local_path and os.path.exists(local_path):
            valid_images.append(image)

    print(f"有效图片: {len(valid_images)} 个")

    if valid_images:
        print("\n前5个有效图片:")
        for i, img in enumerate(valid_images[:5]):
            print(f"  {i+1}. {img.get('original_filename')}")
            print(f"     local_path: {img.get('local_path')}")
    else:
        print("没有有效图片！")
else:
    print(f"获取失败")

# 2. 测试获取视频
print("\n[2] 测试获取视频:")
print("-"*80)
response = requests.get(f"{API_BASE}/videos/", params={'user_id': 1, 'per_page': 20})
print(f"状态码: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    videos = data.get('results', [])
    print(f"获取到 {len(videos)} 个视频")

    # 模拟前端 onVideosLoaded 处理
    valid_videos = []
    for video in videos:
        local_path = video.get('local_path')
        if local_path and os.path.exists(local_path):
            valid_videos.append(video)

    print(f"有效视频: {len(valid_videos)} 个")

    if valid_videos:
        print("\n所有有效视频:")
        for i, vid in enumerate(valid_videos):
            print(f"  {i+1}. {vid.get('original_filename')}")
            print(f"     local_path: {vid.get('local_path')}")
    else:
        print("没有有效视频！")
else:
    print(f"获取失败")

# 3. 模拟前端完整加载流程
print("\n[3] 模拟前端完整加载流程:")
print("-"*80)

class FakeDatabaseManager:
    """模拟前端DatabaseManager"""
    def __init__(self, base_url):
        self.base_url = base_url

    def load_images(self):
        """加载图片"""
        response = requests.get(
            f"{self.base_url}/images/",
            params={'user_id': 1, 'per_page': 1000},
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            return result.get('results', [])
        return []

    def load_videos(self):
        """加载视频"""
        response = requests.get(
            f"{self.base_url}/videos/",
            params={'user_id': 1, 'per_page': 1000},
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            return result.get('results', [])
        return []

db = FakeDatabaseManager(API_BASE)

# 加载图片
print("加载图片...")
images = db.load_images()
print(f"返回 {len(images)} 个图片")

valid_images = []
for image in images:
    local_path = image.get('local_path')
    if local_path and os.path.exists(local_path):
        valid_images.append(image)

print(f"过滤后 {len(valid_images)} 个有效图片")

# 加载视频
print("\n加载视频...")
videos = db.load_videos()
print(f"返回 {len(videos)} 个视频")

valid_videos = []
for video in videos:
    local_path = video.get('local_path')
    if local_path and os.path.exists(local_path):
        valid_videos.append(video)

print(f"过滤后 {len(valid_videos)} 个有效视频")

# 4. 总结
print("\n" + "="*80)
print("[4] 总结:")
print("-"*80)
print(f"图片: {len(valid_images)} 个可用")
print(f"视频: {len(valid_videos)} 个可用")

if len(valid_images) > 0 or len(valid_videos) > 0:
    print("\n状态: 前端应该能够显示图片和视频!")
else:
    print("\n状态: 前端无法显示任何文件!")

print("="*80)
