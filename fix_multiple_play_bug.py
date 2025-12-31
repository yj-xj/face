#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复多次播放视频后速度变快的bug
"""
import os

file_path = "E:/face/frontend/face_swap_ui_enhanced.py"

# 读取文件
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 修复1: 在playWithOpenCV开始时断开旧的信号连接
old_timer_stop = '''            # 检查是否已经有正在运行的播放线程
            if hasattr(self, 'cv_play_timer') and self.cv_play_timer.isActive():
                self.cv_play_timer.stop()'''

new_timer_stop = '''            # 检查是否已经有正在运行的播放线程
            if hasattr(self, 'cv_play_timer') and self.cv_play_timer is not None:
                if self.cv_play_timer.isActive():
                    self.cv_play_timer.stop()
                # 断开旧的信号连接，防止重复触发
                try:
                    self.cv_play_timer.timeout.disconnect()
                except Exception:
                    pass  # 如果没有连接，忽略错误'''

content = content.replace(old_timer_stop, new_timer_stop)

# 修复2: 在连接信号前也断开一次，确保没有重复连接
old_connect = '''            self.cv_play_timer.timeout.connect(self.showNextFrame)

            self.cv_play_timer.start(delay)'''

new_connect = '''            # 确保信号只连接一次
            try:
                self.cv_play_timer.timeout.disconnect()
            except Exception:
                pass
            self.cv_play_timer.timeout.connect(self.showNextFrame)

            self.cv_play_timer.start(delay)'''

content = content.replace(old_connect, new_connect)

# 写回文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] Fixed multiple-play speed bug:")
print("  1. Disconnect old timer signals before creating new ones")
print("  2. Prevent duplicate signal connections")
print("  3. This fixes the issue where playing multiple videos")
print("     causes exponential speed increase")
