#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复playVideoFromList方法
"""
import re

file_path = "E:/face/frontend/face_swap_ui_enhanced.py"

# 读取文件
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 新的playVideoFromList方法
new_method = '''    def playVideoFromList(self, item):
        """双击视频列表项播放视频"""
        try:
            video_path = item.data(Qt.UserRole)

            print(f"[双击播放] 开始处理...")
            print(f"[双击播放] 视频路径: {video_path}")

            if not video_path:
                print("[双击播放] 错误: 视频路径为空")
                QMessageBox.warning(self, "警告", "视频路径为空")
                return

            if not os.path.exists(video_path):
                print(f"[双击播放] 错误: 文件不存在 - {video_path}")
                QMessageBox.warning(self, "警告", f"视频文件不存在:\\n{video_path}")
                return

            filename = os.path.basename(video_path)
            print(f"[双击播放] 文件名: {filename}")
            self.statusBar().showMessage(f"正在播放: {filename}")

            # 设置选中的视频（这样后续处理可以继续使用）
            self.selected_video_path = video_path

            # 自动生成输出路径
            base_name = os.path.splitext(filename)[0]
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output_videos")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{base_name}_face_swap.mp4")
            self.output_path_edit.setText(output_path)

            # 停止当前播放
            print("[双击播放] 停止当前播放...")
            self.stopPlayback()

            # 清空视频标签并重置
            self.cv_video_label.clear()
            self.cv_video_label.setText("<font color='#AAAAAA'>正在加载视频...</font>")
            self.cv_video_label.repaint()

            # 保存当前视频路径
            self.current_video_path = video_path

            print(f"[双击播放] 准备调用playWithOpenCV...")

            # 使用QTimer延迟调用，确保UI更新完成
            QTimer.singleShot(100, self.playWithOpenCV)

        except Exception as e:
            print(f"[双击播放] 异常: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"播放视频时出错:\\n{str(e)}")
'''

# 使用正则表达式替换旧方法
pattern = r'    def playVideoFromList\(self, item\):.*?(?=\n    def )'
replacement = new_method + '\n'

content_new = re.sub(pattern, replacement, content, count=1, flags=re.DOTALL)

# 写回文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content_new)

print("playVideoFromList方法已更新！")
