#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试数据库和前端加载流程
"""
import requests
import os
import json

API_BASE = "http://localhost:8000/api"

print("="*80)
print("调试数据库和前端加载")
print("="*80)

# 1. 测试获取图片
print("\n[1] 获取人脸图片数据:")
print("-"*80)
try:
    response = requests.get(f"{API_BASE}/images/", params={'user_id': 1, 'per_page': 20})
    print(f"状态码: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"返回的数据结构: {list(data.keys())}")
        print(f"count: {data.get('count')}")
        print(f"results长度: {len(data.get('results', []))}")

        # 检查第一个图片的数据结构
        if data.get('results'):
            first_img = data['results'][0]
            print(f"\n第一个图片的所有字段:")
            for key, value in first_img.items():
                if key == 'local_path':
                    print(f"  {key}: {value}")
                    if value and os.path.exists(value):
                        print(f"    -> 文件存在！")
                    else:
                        print(f"    -> 文件不存在！")
                else:
                    print(f"  {key}: {value}")

        # 统计有local_path的图片
        images_with_path = [img for img in data.get('results', []) if img.get('local_path')]
        print(f"\n有local_path的图片数量: {len(images_with_path)}/{len(data.get('results', []))}")

        print(f"\n前5个有路径的图片:")
        for i, img in enumerate(images_with_path[:5]):
            path = img.get('local_path')
            exists = os.path.exists(path) if path else False
            print(f"  {i+1}. {img.get('original_filename')}")
            print(f"     路径: {path}")
            print(f"     存在: {'是' if exists else '否'}")
    else:
        print(f"错误响应: {response.text[:500]}")
except Exception as e:
    print(f"异常: {e}")
    import traceback
    traceback.print_exc()

# 2. 测试获取视频
print("\n" + "="*80)
print("[2] 获取视频数据:")
print("-"*80)
try:
    response = requests.get(f"{API_BASE}/videos/", params={'user_id': 1, 'per_page': 20})
    print(f"状态码: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"count: {data.get('count')}")
        print(f"results长度: {len(data.get('results', []))}")

        # 检查第一个视频
        if data.get('results'):
            first_video = data['results'][0]
            print(f"\n第一个视频的关键字段:")
            print(f"  id: {first_video.get('id')}")
            print(f"  original_filename: {first_video.get('original_filename')}")
            print(f"  local_path: {first_video.get('local_path')}")

            path = first_video.get('local_path')
            if path:
                exists = os.path.exists(path)
                print(f"  文件存在: {'是' if exists else '否'}")

        # 统计有local_path的视频
        videos_with_path = [v for v in data.get('results', []) if v.get('local_path')]
        print(f"\n有local_path的视频数量: {len(videos_with_path)}/{len(data.get('results', []))}")

        print(f"\n所有有路径的视频:")
        for i, video in enumerate(videos_with_path):
            path = video.get('local_path')
            exists = os.path.exists(path) if path else False
            print(f"  {i+1}. {video.get('original_filename')}")
            print(f"     路径: {path}")
            print(f"     存在: {'是' if exists else '否'}")
    else:
        print(f"错误响应: {response.text[:500]}")
except Exception as e:
    print(f"异常: {e}")
    import traceback
    traceback.print_exc()

# 3. 检查前端期望的数据格式
print("\n" + "="*80)
print("[3] 前端期望的数据格式分析:")
print("-"*80)
print("前端 onImagesLoaded 期望: list of dicts")
print("每个dict应该包含:")
print("  - local_path: 本地文件路径")
print("  - original_filename: 原始文件名")
print("  - file: 服务器文件URL (可选)")
print("\n前端 onVideosLoaded 期望: list of dicts")
print("每个dict应该包含:")
print("  - local_path: 本地文件路径")
print("  - original_filename: 原始文件名")
print("  - file: 服务器文件URL (可选)")

# 4. 模拟前端加载流程
print("\n" + "="*80)
print("[4] 模拟前端加载流程:")
print("-"*80)

try:
    response = requests.get(f"{API_BASE}/images/", params={'user_id': 1, 'per_page': 20})
    if response.status_code == 200:
        data = response.json()
        images = data.get('results', [])
        print(f"获取到 {len(images)} 个图片")

        valid_images = []
        for img in images:
            local_path = img.get('local_path')
            if local_path and os.path.exists(local_path):
                valid_images.append(img)
                print(f"  ✓ {img.get('original_filename')} -> {local_path}")
            elif local_path:
                print(f"  ✗ {img.get('original_filename')} -> {local_path} (不存在)")
            else:
                print(f"  ✗ {img.get('original_filename')} -> 无本地路径")

        print(f"\n有效图片数量: {len(valid_images)}")

    response = requests.get(f"{API_BASE}/videos/", params={'user_id': 1, 'per_page': 20})
    if response.status_code == 200:
        data = response.json()
        videos = data.get('results', [])
        print(f"\n获取到 {len(videos)} 个视频")

        valid_videos = []
        for video in videos:
            local_path = video.get('local_path')
            if local_path and os.path.exists(local_path):
                valid_videos.append(video)
                print(f"  ✓ {video.get('original_filename')} -> {local_path}")
            elif local_path:
                print(f"  ✗ {video.get('original_filename')} -> {local_path} (不存在)")
            else:
                print(f"  ✗ {video.get('original_filename')} -> 无本地路径")

        print(f"\n有效视频数量: {len(valid_videos)}")

except Exception as e:
    print(f"异常: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("调试完成")
print("="*80)
