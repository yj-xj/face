#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试前端从数据库加载功能
"""
import sys
import os

# 添加frontend路径
sys.path.insert(0, 'E:/face/frontend')

from database_manager import DatabaseManager

print("="*60)
print("测试前端数据库加载")
print("="*60)

# 创建数据库管理器
db_manager = DatabaseManager(base_url="http://localhost:8000/api")

# 测试连接
print("\n1. 测试服务器连接:")
if db_manager.check_server():
    print("   服务器连接正常")
else:
    print("   服务器连接失败")
    sys.exit(1)

# 测试加载图片
print("\n2. 加载图片:")
images_loader = db_manager.load_images()
if images_loader:
    print("   图片加载线程已启动")

    # 等待加载完成
    import time
    images_loader.wait()
    print("   图片加载完成")
else:
    print("   图片加载失败")

# 测试加载视频
print("\n3. 加载视频:")
videos_loader = db_manager.load_videos()
if videos_loader:
    print("   视频加载线程已启动")

    # 等待加载完成
    videos_loader.wait()
    print("   视频加载完成")
else:
    print("   视频加载失败")

print("\n完成!")
