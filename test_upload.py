#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试本地路径上传功能
"""
import requests
import os
import sys

# 测试上传本地路径
API_BASE = "http://localhost:8000/api"

# 使用一个已存在的本地视频文件进行测试
test_video_path = r"C:\Users\叶俊\Downloads\Video_1701829989285.mp4"

if not os.path.exists(test_video_path):
    print(f"Test file does not exist: {test_video_path}")
    print("Please modify test_video_path to a real file path")
    sys.exit(1)

print(f"Test file: {test_video_path}")
print(f"File size: {os.path.getsize(test_video_path)} bytes")

# 准备上传数据（只发送本地路径，不上传文件内容）
data = {
    'local_path': test_video_path,
    'original_filename': os.path.basename(test_video_path)
}

print("\nUploading local path to backend...")
print(f"API URL: {API_BASE}/videos/")
print(f"Request data: {data}")

try:
    response = requests.post(
        f"{API_BASE}/videos/",
        json=data,
        timeout=30
    )

    print(f"\nResponse status: {response.status_code}")
    print(f"Response content:")

    if response.status_code in [200, 201]:
        result = response.json()
        print(f"  - ID: {result.get('id')}")
        print(f"  - Original filename: {result.get('original_filename')}")
        print(f"  - Local path: {result.get('local_path')}")
        print(f"  - Server file: {result.get('file')}")
        print(f"  - Duration: {result.get('duration')} sec")
        print(f"  - Resolution: {result.get('width')}x{result.get('height')}")
        print(f"  - FPS: {result.get('fps')}")

        if result.get('local_path') == test_video_path:
            print("\n[SUCCESS] Local path saved correctly!")
        else:
            print(f"\n[FAIL] Local path mismatch")
            print(f"   Expected: {test_video_path}")
            print(f"   Actual: {result.get('local_path')}")
    else:
        print(f"  {response.text}")
        print("\n[FAIL] Upload failed")

except Exception as e:
    print(f"\n[ERROR] Request error: {e}")
    import traceback
    traceback.print_exc()

# Test getting video list
print("\n" + "="*50)
print("Testing video list retrieval...")
try:
    response = requests.get(
        f"{API_BASE}/videos/",
        params={'user_id': 1, 'per_page': 10},
        timeout=30
    )

    if response.status_code == 200:
        result = response.json()
        videos = result.get('results', [])

        print(f"Found {len(videos)} videos:")
        for video in videos:
            local_path = video.get('local_path')
            has_local = "Y" if local_path else "N"
            print(f"  [{has_local}] ID:{video.get('id')} - {video.get('original_filename')}")
            if local_path:
                exists = "exists" if os.path.exists(local_path) else "does not exist"
                print(f"      Local path: {local_path} ({exists})")

    else:
        print(f"Retrieval failed: {response.status_code}")

except Exception as e:
    print(f"Request error: {e}")
