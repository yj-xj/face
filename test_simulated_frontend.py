#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模拟前端加载流程测试
"""
import sys
import os

# 添加frontend路径
sys.path.insert(0, 'E:/face/frontend')

print("="*80)
print("模拟前端加载流程")
print("="*80)

# 导入database_manager
from database_manager import DatabaseManager

# 创建数据库管���器
print("\n[1] 创建数据库管理器...")
db_manager = DatabaseManager(base_url="http://localhost:8000/api")

# 检查连接
print("\n[2] 检查服务器连接...")
if db_manager.check_server():
    print("[OK] 服务器连接正常")
else:
    print("[FAIL] 服务器连接失败")
    sys.exit(1)

# 加载图片
print("\n[3] 加载图片...")
image_loader = db_manager.load_images()
if image_loader:
    print("[OK] 图片加载线程已启动")
    # 等待完成
    image_loader.wait()
    print("[OK] 图片加载完成")
else:
    print("[FAIL] 图片加载失败")

# 加载视频
print("\n[4] 加载视频...")
video_loader = db_manager.load_videos()
if video_loader:
    print("[OK] 视频加载线程已启动")
    # 等待完成
    video_loader.wait()
    print("[OK] 视频加载完成")
else:
    print("[FAIL] 视频加载失败")

print("\n" + "="*80)
print("测试完成")
print("="*80)
