import os
import re
import sys

def fix_indentation():
    """修复face_swap.py文件中的缩进问题"""
    file_path = "src/face_swap.py"
    backup_path = "src/face_swap.py.bak"
    
    print("开始修复face_swap.py中的缩进问题...")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在!")
        return False
    
    # 创建备份
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        with open(backup_path, 'w', encoding='utf-8') as file:
            file.write(content)
        
        print(f"已创建备份文件: {backup_path}")
    except Exception as e:
        print(f"创建备份时出错: {e}")
        return False
    
    # 修复缩进问题
    print("开始修复缩进问题...")
    
    # 特定问题修复: 先修复第200行左右的self.set_app_appearance()缩进问题
    content = re.sub(r'(if self\.root is not None:.*?)\n(\s+)(self\.set_app_appearance\(\))', 
                     r'\1\n            \3', 
                     content, flags=re.DOTALL)
    
    # 1. 修复if self.root is not None:后面的缩进问题
    content = re.sub(r'if self\.root is not None:\n(\s+)self\.', 
                    r'if self.root is not None:\n            self.', 
                    content)
    
    # 2. 修复try-except块中的缩进问题
    content = re.sub(r'try:\n(\s+)# 加载并调整图片大小', 
                    r'try:\n                # 加载并调整图片大小', 
                    content)
    
    # 3. 修复insightface_face_swap方法中的缩进问题
    content = re.sub(r'source_faces = self\.face_analyser\.get\(source_rgb\)\n(\s+)if len\(source_faces\) == 0:', 
                    r'source_faces = self.face_analyser.get(source_rgb)\n                if len(source_faces) == 0:', 
                    content)
    
    content = re.sub(r'if len\(source_faces\) == 0:\n(\s+)return frame', 
                    r'if len(source_faces) == 0:\n                    return frame', 
                    content)
    
    # 4. 修复write_frames_in_order方法中的缩进问题
    content = re.sub(r'logger\.error\(f"写入帧 {next_frame_to_write} 时出错: {e}"\)\n(\s+)else:', 
                    r'logger.error(f"写入帧 {next_frame_to_write} 时出错: {e}")\n                else:', 
                    content)
    
    # 5. 修复load_video_player方法中try-except块的缩进问题
    content = re.sub(r'messagebox\.showinfo\("处理完成", "视频处理完成，可以在播放器中查看结果"\)\n(\s+)except Exception as e:', 
                    r'messagebox.showinfo("处理完成", "视频处理完成，可以在播放器中查看结果")\n        except Exception as e:', 
                    content)
    
    # 6. 修复stop_video方法中的缩进问题
    content = re.sub(r'self\.time_label\.config\(text=f"00:00 / {self\.format_time\(duration\)}"\)\n(\s+)else:', 
                    r'self.time_label.config(text=f"00:00 / {self.format_time(duration)}")\n                else:', 
                    content)
    
    # 7. 修复preview_frame方法中的缩进问题
    content = re.sub(r'def process_preview\(\):\n(\s+)try:', 
                    r'def process_preview():\n            try:', 
                    content)
    
    content = re.sub(r'total_frames = int\(cap\.get\(cv2\.CAP_PROP_FRAME_COUNT\)\)\n(\s+)middle_frame = total_frames // 2', 
                    r'total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n                middle_frame = total_frames // 2', 
                    content)
    
    content = re.sub(r'messagebox\.showerror\("错误", "无法读取视频帧"\)\n(\s+)preview_window\.destroy\(\)', 
                    r'messagebox.showerror("错误", "无法读取视频帧")\n                preview_window.destroy()', 
                    content)
    
    content = re.sub(r'progress_label\.config\(text="处理完成"\)\n(\s+)else:', 
                    r'progress_label.config(text="处理完成")\n                else:', 
                    content)
    
    content = re.sub(r'else:\n(\s+)progress_label\.config\(text="处理失败 - 未能检测到人脸或应用替换"\)\n(\s+)except Exception as e:', 
                    r'else:\n                    progress_label.config(text="处理失败 - 未能检测到人脸或应用替换")\n            except Exception as e:', 
                    content)
    
    # 保存修复后的文件
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        
        print(f"缩进问题已修复: {file_path}")
        return True
    except Exception as e:
        print(f"保存修复后的文件时出错: {e}")
        return False

if __name__ == "__main__":
    # 确保输出正确刷新到控制台
    sys.stdout.flush()
    success = fix_indentation()
    print("修复程序执行完成" + (" - 成功!" if success else " - 失败!"))
    sys.stdout.flush() 