#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复视频播放速度和摄像头卡顿问题
"""
import os

file_path = "E:/face/frontend/face_swap_ui_enhanced.py"

# 读取文件
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 修复1: 改进fps验证和delay计算
old_fps_check = '''            # 计算延迟（毫秒）- 修复fps范围限制
            if self.cv_fps <= 0 or self.cv_fps > 240:
                self.cv_fps = 30  # 默认30fps，但允许高帧率视频正常播放
            delay = int(1000 / self.cv_fps / self.playback_speed_factor)
            if delay <= 0: # 防止延迟为0或负数
                delay = 1 # 设置一个最小延迟'''

new_fps_check = '''            # 计算延迟（毫秒）- 改进fps验证
            print(f"[调试] 原始fps: {self.cv_fps:.2f}, 播放速度因子: {self.playback_speed_factor}")
            if self.cv_fps <= 0 or self.cv_fps > 240 or not (10 <= self.cv_fps <= 120):
                print(f"[警告] fps异常({self.cv_fps:.2f})，使用默认值30")
                self.cv_fps = 30  # 默认30fps
            delay = int(1000 / self.cv_fps / self.playback_speed_factor)
            if delay < 10: # 最小10ms，防止播放过快
                delay = 10
            print(f"[调试] 计算delay: {delay}ms (fps={self.cv_fps:.2f}, speed={self.playback_speed_factor}x)")'''

content = content.replace(old_fps_check, new_fps_check)

# 修复2: 优化摄像头处理，进一步降低分辨率
old_camera_process = '''                        try:
                            if self.face_swap_app.inswapper is not None and self.face_swap_app.face_analyser is not None:
                                h, w = frame.shape[:2]
                                process_size = 480  # 降低分辨率以减少卡顿

                                if w > process_size:
                                    scale = process_size / w
                                    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale,
                                                           interpolation=cv2.INTER_AREA)

                                    # 异步处理 - 使用try-catch确保不阻塞
                                    try:
                                        processed_small = self.face_swap_app.insightface_face_swap(small_frame, target_face)
                                        if processed_small is not None:
                                            # 使用快速插值减少延迟
                                            processed_frame = cv2.resize(processed_small, (w, h),
                                                                   interpolation=cv2.INTER_LINEAR)
                                            self.frame_ready.emit(processed_frame)
                                            cached_result = processed_frame
                                        else:
                                            self.frame_ready.emit(frame)
                                    except:
                                        self.frame_ready.emit(frame)'''

new_camera_process = '''                        try:
                            if self.face_swap_app.inswapper is not None and self.face_swap_app.face_analyser is not None:
                                h, w = frame.shape[:2]
                                process_size = 320  # 进一步降低分辨率，优先流畅度

                                if w > process_size:
                                    scale = process_size / w
                                    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale,
                                                           interpolation=cv2.INTER_NEAREST)  # 使用最快插值

                                    # 异步处理 - 使用try-catch确保不阻塞
                                    try:
                                        processed_small = self.face_swap_app.insightface_face_swap(small_frame, target_face)
                                        if processed_small is not None:
                                            # 快速放大
                                            processed_frame = cv2.resize(processed_small, (w, h),
                                                                   interpolation=cv2.INTER_LINEAR)
                                            self.frame_ready.emit(processed_frame)
                                            cached_result = processed_frame
                                        else:
                                            self.frame_ready.emit(frame)
                                    except:
                                        self.frame_ready.emit(frame)'''

content = content.replace(old_camera_process, new_camera_process)

# 修复3: 改变skip_frames策略，改为1（每2帧处理1帧）
old_skip_frames = '''            # 主循环 - 终极优化版 (异步处理 + 多级缓存)
            frame_count = 0
            skip_frames = 2  # 处理每第3帧，平衡流畅度和质量'''

new_skip_frames = '''            # 主循环 - 终极优化版 (异步处理 + 多级缓存)
            frame_count = 0
            skip_frames = 1  # 每处理1帧跳过1帧，更流畅'''

content = content.replace(old_skip_frames, new_skip_frames)

# 写回文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] Fixes applied:")
print("  1. Improved fps validation with debug output")
print("  2. Reduced camera resolution to 320px for smoother performance")
print("  3. Changed skip_frames from 2 to 1 (process every other frame)")
print("  4. Used INTER_NEAREST for faster downscaling")
print("\nPlease test and check the console output for fps/delay values")
