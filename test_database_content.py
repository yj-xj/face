#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试数据库内容
"""
import requests
import os

API_BASE = "http://localhost:8000/api"

print("="*60)
print("测试数据库内容")
print("="*60)

# 测试获取图片
print("\n1. 获取人脸图片:")
try:
    response = requests.get(f"{API_BASE}/images/", params={'user_id': 1, 'per_page': 20})
    if response.status_code == 200:
        result = response.json()
        images = result.get('results', result.get('images', []))
        print(f"共找到 {len(images)} 个图片:")

        for img in images:
            local_path = img.get('local_path')
            original_filename = img.get('original_filename')
            has_local = "Y" if local_path else "N"
            exists = ""

            if local_path and os.path.exists(local_path):
                exists = "✓ 存在"
            elif local_path:
                exists = "✗ 不存在"

            print(f"  [{has_local}] ID:{img.get('id')} - {original_filename}")
            if local_path:
                print(f"      路径: {local_path} {exists}")
    else:
        print(f"获取失败: {response.status_code}")
except Exception as e:
    print(f"错误: {e}")

# 测试获取视频
print("\n2. 获取输入视频:")
try:
    response = requests.get(f"{API_BASE}/videos/", params={'user_id': 1, 'per_page': 20})
    if response.status_code == 200:
        result = response.json()
        videos = result.get('results', result.get('videos', []))
        print(f"共找到 {len(videos)} 个视频:")

        for video in videos:
            local_path = video.get('local_path')
            original_filename = video.get('original_filename')
            has_local = "Y" if local_path else "N"
            exists = ""

            if local_path and os.path.exists(local_path):
                exists = "✓ 存在"
            elif local_path:
                exists = "✗ 不存在"

            print(f"  [{has_local}] ID:{video.get('id')} - {original_filename}")
            if local_path:
                print(f"      路径: {local_path} {exists}")
    else:
        print(f"获取失败: {response.status_code}")
except Exception as e:
    print(f"错误: {e}")

# 统计信息
print("\n" + "="*60)
print("统计信息:")
print("="*60)

try:
    response = requests.get(f"{API_BASE}/images/")
    if response.status_code == 200:
        result = response.json()
        count = result.get('count', len(result.get('results', [])))
        with_local = sum(1 for img in result.get('results', []) if img.get('local_path'))
        print(f"图片: {count} 个 (包含本地路径: {with_local} 个)")

    response = requests.get(f"{API_BASE}/videos/")
    if response.status_code == 200:
        result = response.json()
        count = result.get('count', len(result.get('results', [])))
        with_local = sum(1 for vid in result.get('results', []) if vid.get('local_path'))
        print(f"视频: {count} 个 (包含本地路径: {with_local} 个)")
except Exception as e:
    print(f"统计失败: {e}")

print("\n完成!")
