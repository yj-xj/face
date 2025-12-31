#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试增强版UI的启动和基本功能
"""
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt5.QtWidgets import QApplication
from face_swap_ui_enhanced import EnhancedFaceSwapUI, AppMode

def test_ui_startup():
    """测试UI启动"""
    print("=" * 60)
    print("测试增强版UI启动")
    print("=" * 60)

    app = QApplication(sys.argv)

    try:
        # 创建主窗口
        window = EnhancedFaceSwapUI()
        print("[OK] 主窗口创建成功")

        # 检查模式
        print(f"[OK] 当前模式: {window.current_mode}")
        print(f"[OK] 摄像头线程状态: {window.camera_thread}")

        # 检查UI组件
        assert hasattr(window, 'video_mode_btn'), "缺少视频模式按钮"
        assert hasattr(window, 'camera_mode_btn'), "缺少摄像头模式按钮"
        assert hasattr(window, 'camera_toggle_btn'), "缺少摄像头开关按钮"
        assert hasattr(window, 'snapshot_btn'), "缺少拍照按钮"
        assert hasattr(window, 'camera_processing_check'), "缺少处理开关"
        print("[OK] 所有UI组件检查通过")

        # 显示窗口
        window.show()
        print("[OK] 窗口显示成功")

        print("\n" + "=" * 60)
        print("所有测试通过！")
        print("=" * 60)
        print("\n提示:")
        print("1. 点击 '摄像头模式' 按钮切换到摄像头模式")
        print("2. 选择一张人脸图片")
        print("3. 点击 '开启摄像头' 开始实时换脸")
        print("4. 使用 '拍照' 按钮保存当前帧")
        print("5. 切换回 '视频模式' 进行视频处理")

        sys.exit(app.exec_())

    except Exception as e:
        print(f"\n[ERROR] 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_ui_startup()

