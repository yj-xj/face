import sys
import os
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QProgressBar, QFrame, QSlider, QStyle)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor
import numpy as np
from PIL import Image
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoThread(QThread):
    frame_ready = pyqtSignal(QImage)
    
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.running = True
        
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        while self.running:
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.frame_ready.emit(qt_image)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap.release()
        
    def stop(self):
        self.running = False
        self.wait()

class FaceSwapApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI人脸替换系统")
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QWidget {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            QPushButton {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                padding: 8px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
                border: 2px solid #4d4d4d;
            }
            QPushButton:pressed {
                background-color: #4d4d4d;
            }
            QProgressBar {
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                text-align: center;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background-color: #00a8ff;
                border-radius: 3px;
            }
            QLabel {
                color: #ffffff;
            }
            QSlider::groove:horizontal {
                border: 1px solid #3d3d3d;
                height: 8px;
                background: #2d2d2d;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00a8ff;
                border: 1px solid #00a8ff;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """)
        
        # 初始化变量
        self.video_path = None
        self.face_image_path = None
        self.video_thread = None
        
        # 创建主窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建主布局
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # 创建左侧视频播放区域
        self.create_video_player()
        
        # 创建右侧控制区域
        self.create_control_panel()
        
        # 设置窗口大小
        self.resize(1200, 800)
        
    def create_video_player(self):
        # 创建左侧视频播放区域
        video_frame = QFrame()
        video_frame.setFrameStyle(QFrame.StyledPanel)
        video_frame.setStyleSheet("""
            QFrame {
                background-color: #000000;
                border: 2px solid #3d3d3d;
                border-radius: 10px;
            }
        """)
        
        video_layout = QVBoxLayout(video_frame)
        
        # 视频显示标签
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        video_layout.addWidget(self.video_label)
        
        # 视频控制按钮
        control_layout = QHBoxLayout()
        
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.toggle_video)
        control_layout.addWidget(self.play_button)
        
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop_video)
        control_layout.addWidget(self.stop_button)
        
        video_layout.addLayout(control_layout)
        
        self.main_layout.addWidget(video_frame, stretch=2)
        
    def create_control_panel(self):
        # 创建右侧控制面板
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.StyledPanel)
        control_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 2px solid #3d3d3d;
                border-radius: 10px;
                padding: 20px;
            }
        """)
        
        control_layout = QVBoxLayout(control_frame)
        
        # 添加标题
        title_label = QLabel("控制面板")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                color: #00a8ff;
                padding: 10px;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(title_label)
        
        # 添加分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #3d3d3d;")
        control_layout.addWidget(line)
        
        # 添加文件选择按钮
        self.select_video_btn = QPushButton("选择视频文件")
        self.select_video_btn.clicked.connect(self.select_video)
        control_layout.addWidget(self.select_video_btn)
        
        self.select_face_btn = QPushButton("选择人脸图片")
        self.select_face_btn.clicked.connect(self.select_face_image)
        control_layout.addWidget(self.select_face_btn)
        
        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        control_layout.addWidget(self.progress_bar)
        
        # 添加处理按钮
        self.process_btn = QPushButton("开始处理")
        self.process_btn.clicked.connect(self.process_video)
        control_layout.addWidget(self.process_btn)
        
        # 添加状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.status_label)
        
        # 添加弹性空间
        control_layout.addStretch()
        
        self.main_layout.addWidget(control_frame, stretch=1)
        
    def select_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", 
                                                 "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*.*)")
        if file_name:
            self.video_path = file_name
            self.status_label.setText(f"已选择视频: {os.path.basename(file_name)}")
            
    def select_face_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择人脸图片", "", 
                                                 "图片文件 (*.jpg *.jpeg *.png);;所有文件 (*.*)")
        if file_name:
            self.face_image_path = file_name
            self.status_label.setText(f"已选择人脸图片: {os.path.basename(file_name)}")
            
    def toggle_video(self):
        if self.video_thread is None or not self.video_thread.running:
            if self.video_path:
                self.video_thread = VideoThread(self.video_path)
                self.video_thread.frame_ready.connect(self.update_frame)
                self.video_thread.start()
                self.play_button.setText("暂停")
            else:
                self.status_label.setText("请先选择视频文件")
        else:
            self.video_thread.stop()
            self.play_button.setText("播放")
            
    def stop_video(self):
        if self.video_thread:
            self.video_thread.stop()
            self.play_button.setText("播放")
            self.video_label.clear()
            
    def update_frame(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
    def process_video(self):
        if not self.video_path or not self.face_image_path:
            self.status_label.setText("请先选择视频和人脸图片")
            return
            
        self.status_label.setText("正在处理...")
        self.progress_bar.setValue(0)
        # TODO: 实现视频处理逻辑
        
    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceSwapApp()
    window.show()
    sys.exit(app.exec_())

def main():
    """程序入口函数"""
    app = QApplication(sys.argv)
    window = FaceSwapApp()
    window.show()
    sys.exit(app.exec_()) 