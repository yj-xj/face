#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复skip_frames逻辑错误
"""
import os

file_path = "E:/face/frontend/face_swap_ui_enhanced.py"

# 读取文件
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 修复skip_frames逻辑
old_logic = '''                # 分离处理和显示逻辑
                if self.processing_enabled and target_face is not None:
                    if frame_count % skip_frames == 0:'''

new_logic = '''                # 分离处理和显示逻辑
                if self.processing_enabled and target_face is not None:
                    if frame_count % (skip_frames + 1) == 0:  # 修复：skip_frames=1时每2帧处理1帧'''

content = content.replace(old_logic, new_logic)

# 写回文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] Fixed skip_frames logic bug:")
print("  - Changed 'frame_count % skip_frames == 0'")
print("  - To 'frame_count % (skip_frames + 1) == 0'")
print("  - Now skip_frames=1 means process every 2 frames (not every frame)")
