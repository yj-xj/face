#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试上传API
"""
import requests
import os

def test_image_upload():
    """测试图片上传"""
    api_url = "http://localhost:8000/api/images/"

    # 选择测试图片
    test_image = r"E:\face\data\input_faces\image.png"

    if not os.path.exists(test_image):
        print(f"测试图片不存在: {test_image}")
        return False

    print(f"上传图片: {test_image}")

    try:
        with open(test_image, 'rb') as f:
            files = {'file': ('test_image.png', f, 'image/png')}
            response = requests.post(api_url, files=files, timeout=30)

        print(f"状态码: {response.status_code}")
        if response.status_code in [200, 201]:
            result = response.json()
            print(f"上传成功!")
            print(f"  ID: {result.get('id')}")
            print(f"  文件名: {result.get('original_filename')}")
            print(f"  尺寸: {result.get('width')}x{result.get('height')}")
            print(f"  大小: {result.get('file_size')} bytes")
            return True
        else:
            print(f"上传失败: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_video_upload():
    """测试视频上传"""
    api_url = "http://localhost:8000/api/videos/"

    # 选择测试视频
    test_video = r"E:\face\data\input_videos\Video_1728365311307.mp4"

    if not os.path.exists(test_video):
        print(f"测试视频不存在: {test_video}")
        return False

    print(f"\n上传视频: {test_video}")

    try:
        with open(test_video, 'rb') as f:
            files = {'file': (os.path.basename(test_video), f, 'video/mp4')}
            response = requests.post(api_url, files=files, timeout=120)

        print(f"状态码: {response.status_code}")
        if response.status_code in [200, 201]:
            result = response.json()
            print(f"上传成功!")
            print(f"  ID: {result.get('id')}")
            print(f"  文件名: {result.get('original_filename')}")
            print(f"  尺寸: {result.get('width')}x{result.get('height')}")
            print(f"  时长: {result.get('duration'):.2f}s")
            return True
        else:
            print(f"上传失败: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_get_images():
    """测试获取图片列表"""
    api_url = "http://localhost:8000/api/images/"

    print(f"\n获取图片列表...")

    try:
        response = requests.get(api_url, timeout=10)
        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            count = result.get('count', 0)
            print(f"成功获取 {count} 张图片")
            return True
        else:
            print(f"获取失败: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"错误: {e}")
        return False

if __name__ == '__main__':
    print("="*50)
    print("测试后端API")
    print("="*50)

    # 测试获取列表
    test_get_images()

    # 测试图片上传
    test_image_upload()

    # 测试视频上传
    test_video_upload()

    print("\n" + "="*50)
    print("测试完成")
    print("="*50)
