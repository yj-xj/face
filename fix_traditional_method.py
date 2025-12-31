#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复传统三角剖分换脸功能
"""
import os

file_path = "E:/face/frontend/face_swap_ui_enhanced.py"

# 读取文件
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 修复1: 在初始化original_app时，确保predictor被正确初始化
old_init = '''        # 初始化原始FaceSwapApp的实例
        self.original_app = OriginalFaceSwapApp(None)
        # 由于原始应用需要root参数，在这里手动设置一些关键属性
        self.original_app.root = None  # 明确设置为None
        # 初始化必要的变量
        self.original_app.swap_method_var = "advanced"  # 默认使用高级替换方法
        self.original_app.color_correction_var = True   # 默认启用颜色校正
        self.original_app.multi_scale_var = True       # 默认使用多尺度检测
        self.original_app.detector_var = "dlib"        # 默认使用dlib检测器
        # 根据模型是否可用来动态设置默认方法
        self.original_app.swapper_var = "inswapper" if (self.original_app.inswapper is not None and self.original_app.face_analyser is not None) else "traditional"
        self.original_app.smoothing_var = 50           # 默认平滑度'''

new_init = '''        # 初始化原始FaceSwapApp的实例
        self.original_app = OriginalFaceSwapApp(None)
        # 由于原始应用需要root参数，在这里手动设置一些关键属性
        self.original_app.root = None  # 明确设置为None
        # 初始化必要的变量
        self.original_app.swap_method_var = "advanced"  # 默认使用高级替换方法
        self.original_app.color_correction_var = True   # 默认启用颜色校正
        self.original_app.multi_scale_var = True       # 默认使用多尺度检测
        self.original_app.detector_var = "dlib"        # 默认使用dlib检测器
        # 根据模型是否可用来动态设置默认方法
        self.original_app.swapper_var = "inswapper" if (self.original_app.inswapper is not None and self.original_app.face_analyser is not None) else "traditional"
        self.original_app.smoothing_var = 50           # 默认平滑度

        # 确保传统方法的predictor被初始化
        if self.original_app.predictor is None and os.path.exists(self.original_app.predictor_path):
            try:
                import dlib
                self.original_app.predictor = dlib.shape_predictor(self.original_app.predictor_path)
                print(f"[初始化] 成功加载传统方法predictor: {self.original_app.predictor_path}")
            except Exception as e:
                print(f"[警告] 初始化predictor失败: {e}")'''

content = content.replace(old_init, new_init)

# 写回文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] Traditional method fix completed:")
print("  1. Ensure predictor is loaded during initialization")
print("  2. Add error handling for predictor initialization")
