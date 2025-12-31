#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试本地路径上传功能
"""
import requests
import os
import sys

# 添加后端路径
sys.path.insert(0, 'E:/face/backend-django')

# 测试上传本地路径
API_BASE = "http://localhost:8000/api"

# 使用一个已存在的本地视频文件进行测试
test_video_path = r"C:\Users\叶俊\Downloads\Video_1701829989285.mp4"

if not os.path.exists(test_video_path):
    print(f"测试文件不存在: {test_video_path}")
    print("请修改脚本中的test_video_path为一个真实存在的文件路径")
    sys.exit(1)

print(f"测试文件: {test_video_path}")
print(f"文件大小: {os.path.getsize(test_video_path)} 字节")

# 准备上传数据（只发送本地路径，不上传文件内容）
data = {
    'local_path': test_video_path,
    'original_filename': os.path.basename(test_video_path)
}

print("\n正在上传本地路径到后端...")
print(f"API URL: {API_BASE}/videos/")
print(f"请求数据: {data}")

try:
    response = requests.post(
        f"{API_BASE}/videos/",
        json=data,
        timeout=30
    )

    print(f"\n响应状态码: {response.status_code}")
    print(f"响应内容:")

    if response.status_code in [200, 201]:
        result = response.json()
        print(f"  - ID: {result.get('id')}")
        print(f"  - 原始文件名: {result.get('original_filename')}")
        print(f"  - 本地路径: {result.get('local_path')}")
        print(f"  - 服务器文件: {result.get('file')}")
        print(f"  - 时长: {result.get('duration')} 秒")
        print(f"  - 分辨率: {result.get('width')}x{result.get('height')}")
        print(f"  - FPS: {result.get('fps')}")

        if result.get('local_path') == test_video_path:
            print("\n✅ 测试成功！本地路径已正确保存")
        else:
            print(f"\n❌ 测试失败！本地路径不匹配")
            print(f"   期望: {test_video_path}")
            print(f"   实际: {result.get('local_path')}")
    else:
        print(f"  {response.text}")
        print("\n❌ 上传失败")

except Exception as e:
    print(f"\n❌ 请求出错: {e}")
    import traceback
    traceback.print_exc()

# 测试获取视频列表
print("\n" + "="*50)
print("测试获取视频列表...")
try:
    response = requests.get(
        f"{API_BASE}/videos/",
        params={'user_id': 1, 'per_page': 10},
        timeout=30
    )

    if response.status_code == 200:
        result = response.json()
        videos = result.get('results', [])

        print(f"共找到 {len(videos)} 个视频:")
        for video in videos:
            local_path = video.get('local_path')
            has_local = "✓" if local_path else "✗"
            print(f"  [{has_local}] ID:{video.get('id')} - {video.get('original_filename')}")
            if local_path:
                exists = "存在" if os.path.exists(local_path) else "不存在"
                print(f"      本地路径: {local_path} ({exists})")

    else:
        print(f"获取失败: {response.status_code}")

except Exception as e:
    print(f"请求出错: {e}")
