#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化摄像头处理线程 - 解决延迟和模糊问题
"""
import re

file_path = "E:/face/frontend/face_swap_ui_enhanced.py"

# 读取文件
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 替换1: 提高处理分辨率，从320提升到640
old_process_size = "process_size = 320  # 极小分辨率，速度优先"
new_process_size = "process_size = 640  # 平衡分辨率和质量"

content = content.replace(old_process_size, new_process_size)

# 替换2: 改进插值方法，从INTER_NEAREST改为INTER_LINEAR
old_resize_1 = '''                                    scale = process_size / w
                                    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale,
                                                           interpolation=cv2.INTER_NEAREST)'''

new_resize_1 = '''                                    scale = process_size / w
                                    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale,
                                                           interpolation=cv2.INTER_AREA)'''

content = content.replace(old_resize_1, new_resize_1)

# 替换3: 改进放大插值方法
old_resize_2 = '''                                    # 快速放大
                                    processed_frame = cv2.resize(processed_small, (w, h),
                                                                   interpolation=cv2.INTER_NEAREST)'''

new_resize_2 = '''                                    # 高质量放大
                                    processed_frame = cv2.resize(processed_small, (w, h),
                                                                   interpolation=cv2.INTER_LINEAR)'''

content = content.replace(old_resize_2, new_resize_2)

# 写回文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("摄像头处理已优化:")
print("- 处理分辨率: 320 -> 640")
print("- 缩小插值: INTER_NEAREST -> INTER_AREA")
print("- 放大插值: INTER_NEAREST -> INTER_LINEAR")
