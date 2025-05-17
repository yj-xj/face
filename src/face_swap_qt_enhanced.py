import sys
import os
import cv2
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QProgressBar, QFrame, QSlider, QStyle, QComboBox,
                            QScrollArea, QGridLayout, QGroupBox, QCheckBox,
                            QSplitter, QStackedWidget, QDialog, QSpacerItem,
                            QSizePolicy, QMenu, QToolButton, QAction, QLineEdit)
from PyQt5.QtCore import (Qt, QThread, pyqtSignal, QTimer, QSize, QUrl, 
                          QPropertyAnimation, QEasingCurve, QRect, QPoint)
from PyQt5.QtGui import (QImage, QPixmap, QPalette, QColor, QFont, 
                        QCursor, QIcon, QRadialGradient, QLinearGradient,
                        QPainter, QPen, QBrush, QFontDatabase)
from PIL import Image
import logging
import pyqtdarktheme as qdarktheme

# 配置日志
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "logs", f"face_swap_{time.strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("face_swap_qt")

# 尝试导入insightface库
try:
    import insightface
    import onnxruntime
    INSIGHTFACE_AVAILABLE = True
    logger.info("成功导入InsightFace模块")
except ImportError as e:
    INSIGHTFACE_AVAILABLE = False
    logger.error(f"导入InsightFace模块时出错: {e}")

class VideoThread(QThread):
    """视频播放线程，从视频文件读取帧并发射信号更新UI"""
    frame_ready = pyqtSignal(QImage)
    duration_updated = pyqtSignal(float)
    position_updated = pyqtSignal(float)
    
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.running = True
        self.paused = False
        self.position = 0
        self.total_frames = 0
        self.fps = 0
        
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = self.total_frames / self.fps if self.fps > 0 else 0
        self.duration_updated.emit(duration)
        
        frame_time = 1000 / self.fps if self.fps > 0 else 33  # 默认30fps
        
        while self.running:
            if not self.paused:
                ret, frame = cap.read()
                if ret:
                    # 更新当前位置
                    self.position = cap.get(cv2.CAP_PROP_POS_FRAMES) / self.total_frames * duration
                    self.position_updated.emit(self.position)
                    
                    # 转换并发送帧
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.frame_ready.emit(qt_image)
                    
                    # 按播放速度延时
                    self.msleep(int(frame_time))
                else:
                    # 播放结束，重新开始
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.position = 0
                    self.position_updated.emit(0)
            else:
                # 暂停状态下，减少CPU使用
                self.msleep(100)
                
        cap.release()
        
    def toggle_pause(self):
        self.paused = not self.paused
        
    def seek(self, position_seconds):
        if position_seconds < 0:
            position_seconds = 0
        
        # 计算帧位置并跳转
        frame_number = int(position_seconds * self.fps)
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            # 更新显示
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_ready.emit(qt_image)
            
            # 更新位置
            duration = self.total_frames / self.fps if self.fps > 0 else 0
            self.position = frame_number / self.total_frames * duration
            self.position_updated.emit(self.position)
        cap.release()
        
    def stop(self):
        self.running = False
        self.wait()

class CircularProgressBar(QWidget):
    """圆形进度条组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0
        self.width = 200
        self.height = 200
        self.progress_width = 10
        self.setMinimumSize(self.width, self.height)
        
        # 设置抗锯齿
        self.setMinimumSize(QSize(self.width, self.height))
        
        # 加载字体
        QFontDatabase.addApplicationFont(":/fonts/Roboto-Medium.ttf")
        
    def setValue(self, value):
        self.value = value
        self.update()
        
    def paintEvent(self, event):
        width = self.width - 4
        height = self.height - 4
        margin = self.progress_width / 2
        value = self.value * 360 / 100
        
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        
        # 创建圆角矩形路径
        rect = QRect(margin, margin, width - margin * 2, height - margin * 2)
        
        # 绘制背景圆
        pen = QPen()
        pen.setWidth(self.progress_width)
        pen.setColor(QColor(68, 68, 68))
        painter.setPen(pen)
        painter.drawArc(rect, 0, 360 * 16)
        
        # 绘制进度圆弧
        pen.setColor(QColor(33, 150, 243))
        painter.setPen(pen)
        painter.drawArc(rect, 90 * 16, -value * 16)
        
        # 绘制文本
        font = QFont("Roboto", 20, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(rect, Qt.AlignCenter, f"{int(self.value)}%")
        
        painter.end()

class GlowingButton(QPushButton):
    """带有发光效果的按钮"""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setMinimumHeight(40)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        
        # 设置圆角和初始样式
        self.setStyleSheet("""
            GlowingButton {
                background-color: #1e88e5;
                color: white;
                border-radius: 20px;
                font-weight: bold;
                padding: 8px 16px;
            }
            GlowingButton:hover {
                background-color: #42a5f5;
            }
            GlowingButton:pressed {
                background-color: #0d47a1;
            }
        """)
        
        # 创建发光动画
        self.animation = QPropertyAnimation(self, b"styleSheet")
        self.animation.setDuration(1000)
        self.animation.setLoopCount(-1)  # 无限循环
        
        # 设置动画步骤
        self.animation.setStartValue("""
            GlowingButton {
                background-color: #1e88e5;
                color: white;
                border-radius: 20px;
                font-weight: bold;
                padding: 8px 16px;
                border: 2px solid #1e88e5;
            }
            GlowingButton:hover {
                background-color: #42a5f5;
            }
            GlowingButton:pressed {
                background-color: #0d47a1;
            }
        """)
        
        self.animation.setEndValue("""
            GlowingButton {
                background-color: #1e88e5;
                color: white;
                border-radius: 20px;
                font-weight: bold;
                padding: 8px 16px;
                border: 2px solid #90caf9;
            }
            GlowingButton:hover {
                background-color: #42a5f5;
            }
            GlowingButton:pressed {
                background-color: #0d47a1;
            }
        """)
        
        self.animation.start()

# 实现主应用程序窗口类
class FaceSwapApp(QMainWindow):
    """主应用程序窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI人脸替换系统")
        
        # 自定义系统外观
        self.set_app_appearance()
        
        # 初始化变量
        self.video_path = None
        self.face_image_path = None
        self.video_thread = None
        self.process_thread = None
        self.output_path = None
        self.models_loaded = False
        
        # 设置默认输出目录
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(self.base_dir, "output_videos")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建主窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建主布局
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(10)
        
        # 创建标题栏
        self.create_title_bar()
        
        # 创建滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        
        # 创建滚动区域内容
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(20)
        
        # 创建主区域的QSplitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.scroll_layout.addWidget(self.main_splitter)
        
        # 创建左侧视频播放区域
        self.create_video_player()
        
        # 创建右侧控制区域
        self.create_control_panel()
        
        # 设置分割器比例
        self.main_splitter.setSizes([int(self.width() * 0.7), int(self.width() * 0.3)])
        
        # 设置滚动区域内容
        self.scroll_area.setWidget(self.scroll_content)
        self.main_layout.addWidget(self.scroll_area)
        
        # 创建状态栏
        self.create_status_bar()
        
        # 设置窗口大小
        self.resize(1280, 800)
        
    def set_app_appearance(self):
        """设置应用程序全局外观"""
        # 应用暗色主题
        qdarktheme.setup_theme(
            theme="dark",
            corner_shape="rounded",
            custom_colors={
                "[dark]": {
                    "primary": "#2196f3",
                    "background": "#121212",
                    "foreground": "#fafafa",
                    "accent": "#00e5ff",
                }
            }
        )
        
        # 自定义额外样式
        extra_styles = """
            QMainWindow {
                background-color: #121212;
            }
            QWidget {
                color: #fafafa;
            }
            QFrame.card {
                background-color: #1d1d1d;
                border-radius: 10px;
                border: 1px solid #333333;
            }
            QLabel.title {
                font-size: 18px;
                font-weight: bold;
                color: #2196f3;
            }
            QLabel.subtitle {
                font-size: 14px;
                color: #03a9f4;
            }
            QPushButton {
                background-color: #2196f3;
                color: white;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #42a5f5;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            QProgressBar {
                border: none;
                text-align: center;
                background-color: #2d2d2d;
                border-radius: 5px;
                height: 10px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2196f3, stop:1 #03a9f4);
                border-radius: 5px;
            }
            QSlider::groove:horizontal {
                border: none;
                height: 6px;
                background-color: #2d2d2d;
                margin: 0px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background-color: #2196f3;
                border: none;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::add-page:horizontal {
                background-color: #2d2d2d;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background-color: #03a9f4;
                border-radius: 3px;
            }
            QComboBox {
                background-color: #2d2d2d;
                color: white;
                border-radius: 5px;
                border: 1px solid #3d3d3d;
                padding: 5px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 24px;
                border-left-width: 1px;
                border-left-color: #3d3d3d;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            QCheckBox {
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid #3d3d3d;
                background-color: #1d1d1d;
            }
            QCheckBox::indicator:checked {
                background-color: #2196f3;
                border: 2px solid #2196f3;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #1d1d1d;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #3d3d3d;
                min-height: 30px;
                border-radius: 6px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #4d4d4d;
            }
            QScrollBar::handle:vertical:pressed {
                background-color: #2196f3;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """
        
        self.setStyleSheet(qdarktheme.load_stylesheet() + extra_styles)
        
    def create_title_bar(self):
        """创建标题栏"""
        title_frame = QFrame()
        title_frame.setObjectName("titleBar")
        title_frame.setStyleSheet("""
            #titleBar {
                background-color: #101010;
                border-radius: 10px;
                padding: 10px;
                margin-bottom: 10px;
            }
        """)
        
        title_layout = QHBoxLayout(title_frame)
        title_layout.setContentsMargins(15, 10, 15, 10)
        
        # 创建标题标签
        title_label = QLabel("AI人脸替换系统")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #03a9f4;
        """)
        
        # 创建副标题标签
        subtitle_label = QLabel("InsightFace高级版")
        subtitle_label.setStyleSheet("""
            font-size: 14px;
            color: #90caf9;
        """)
        
        # 添加到布局
        title_inner_layout = QVBoxLayout()
        title_inner_layout.addWidget(title_label)
        title_inner_layout.addWidget(subtitle_label)
        
        title_layout.addLayout(title_inner_layout)
        title_layout.addStretch()
        
        # 添加到主布局
        self.main_layout.addWidget(title_frame)
        
    def create_video_player(self):
        """创建视频播放区域"""
        # 创建视频播放框架
        video_frame = QFrame()
        video_frame.setObjectName("videoPlayer")
        video_frame.setStyleSheet("""
            #videoPlayer {
                background-color: #0d0d0d;
                border: 1px solid #333333;
                border-radius: 10px;
            }
        """)
        
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(10, 10, 10, 10)
        video_layout.setSpacing(10)
        
        # 创建视频标题
        video_title = QLabel("视频预览")
        video_title.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #03a9f4;
            margin-bottom: 5px;
        """)
        video_title.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(video_title)
        
        # 创建视频显示区域
        self.video_container = QFrame()
        self.video_container.setStyleSheet("""
            background-color: #000000;
            border-radius: 5px;
        """)
        
        # 视频显示标签
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background-color: #000000;")
        
        video_container_layout = QVBoxLayout(self.video_container)
        video_container_layout.setContentsMargins(0, 0, 0, 0)
        video_container_layout.addWidget(self.video_label)
        
        video_layout.addWidget(self.video_container)
        
        # 创建视频控制栏
        control_frame = QFrame()
        control_frame.setObjectName("videoControl")
        control_frame.setStyleSheet("""
            #videoControl {
                background-color: #1a1a1a;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        
        control_layout = QVBoxLayout(control_frame)
        control_layout.setContentsMargins(10, 5, 10, 5)
        control_layout.setSpacing(5)
        
        # 创建时间滑块
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 100)
        self.time_slider.setValue(0)
        self.time_slider.sliderMoved.connect(self.seek_video)
        control_layout.addWidget(self.time_slider)
        
        # 创建时间显示
        time_layout = QHBoxLayout()
        self.current_time = QLabel("00:00")
        self.total_time = QLabel("00:00")
        
        time_layout.addWidget(self.current_time)
        time_layout.addStretch()
        time_layout.addWidget(self.total_time)
        control_layout.addLayout(time_layout)
        
        # 创建播放控制按钮
        buttons_layout = QHBoxLayout()
        
        # 播放按钮
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.setFixedSize(36, 36)
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #2196f3;
                border-radius: 18px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #42a5f5;
            }
        """)
        self.play_button.setIconSize(QSize(20, 20))
        self.play_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.play_button.clicked.connect(self.toggle_video)
        
        # 停止按钮
        self.stop_button = QPushButton()
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.setFixedSize(36, 36)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                border-radius: 18px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #ef5350;
            }
        """)
        self.stop_button.setIconSize(QSize(20, 20))
        self.stop_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.stop_button.clicked.connect(self.stop_video)
        
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.play_button)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addStretch()
        
        control_layout.addLayout(buttons_layout)
        video_layout.addWidget(control_frame)
        
        # 添加到主分割器
        self.main_splitter.addWidget(video_frame)
        
    def create_control_panel(self):
        """创建控制面板"""
        # 创建控制面板框架
        control_frame = QFrame()
        control_frame.setObjectName("controlPanel")
        control_frame.setStyleSheet("""
            #controlPanel {
                background-color: #1a1a1a;
                border: 1px solid #333333;
                border-radius: 10px;
            }
        """)
        
        control_layout = QVBoxLayout(control_frame)
        control_layout.setContentsMargins(15, 15, 15, 15)
        control_layout.setSpacing(15)
        
        # 创建控制面板标题
        control_title = QLabel("控制面板")
        control_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #03a9f4;")
        control_title.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(control_title)
        
        # 创建文件选择区域
        file_group = QGroupBox("文件选择")
        file_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #333333;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                color: #03a9f4;
                padding: 0 5px;
            }
        """)
        
        file_layout = QGridLayout(file_group)
        file_layout.setContentsMargins(10, 15, 10, 10)
        file_layout.setSpacing(10)
        
        # 视频文件选择
        video_label = QLabel("视频文件：")
        self.video_path_label = QLabel("未选择")
        self.video_path_label.setWordWrap(True)
        self.video_path_label.setStyleSheet("color: #bbbbbb;")
        
        self.select_video_btn = QPushButton("选择视频")
        self.select_video_btn.clicked.connect(self.select_video)
        
        file_layout.addWidget(video_label, 0, 0)
        file_layout.addWidget(self.video_path_label, 0, 1)
        file_layout.addWidget(self.select_video_btn, 1, 0, 1, 2)
        
        # 人脸图片选择
        face_label = QLabel("人脸图片：")
        self.face_path_label = QLabel("未选择")
        self.face_path_label.setWordWrap(True)
        self.face_path_label.setStyleSheet("color: #bbbbbb;")
        
        self.select_face_btn = QPushButton("选择人脸")
        self.select_face_btn.clicked.connect(self.select_face_image)
        
        file_layout.addWidget(face_label, 2, 0)
        file_layout.addWidget(self.face_path_label, 2, 1)
        file_layout.addWidget(self.select_face_btn, 3, 0, 1, 2)
        
        # 输出文件选择
        output_label = QLabel("输出文件：")
        self.output_path_label = QLabel("未选择")
        self.output_path_label.setWordWrap(True)
        self.output_path_label.setStyleSheet("color: #bbbbbb;")
        
        self.select_output_btn = QPushButton("选择输出")
        self.select_output_btn.clicked.connect(self.select_output)
        
        file_layout.addWidget(output_label, 4, 0)
        file_layout.addWidget(self.output_path_label, 4, 1)
        file_layout.addWidget(self.select_output_btn, 5, 0, 1, 2)
        
        control_layout.addWidget(file_group)
        
        # 创建高级选项区域
        options_group = QGroupBox("高级选项")
        options_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #333333;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                color: #03a9f4;
                padding: 0 5px;
            }
        """)
        
        options_layout = QVBoxLayout(options_group)
        options_layout.setContentsMargins(10, 15, 10, 10)
        options_layout.setSpacing(10)
        
        # 模型选择
        model_layout = QHBoxLayout()
        model_label = QLabel("替换模型：")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["InsightFace", "传统算法"])
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        
        # 质量选择
        quality_layout = QHBoxLayout()
        quality_label = QLabel("输出质量：")
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["高质量", "标准", "快速处理"])
        
        quality_layout.addWidget(quality_label)
        quality_layout.addWidget(self.quality_combo)
        
        # 调整选项
        self.smooth_faces = QCheckBox("面部平滑处理")
        self.color_correction = QCheckBox("颜色校正")
        self.enhance_quality = QCheckBox("增强输出质量")
        
        # 默认选中
        self.color_correction.setChecked(True)
        
        options_layout.addLayout(model_layout)
        options_layout.addLayout(quality_layout)
        options_layout.addWidget(self.smooth_faces)
        options_layout.addWidget(self.color_correction)
        options_layout.addWidget(self.enhance_quality)
        
        control_layout.addWidget(options_group)
        
        # 创建处理按钮
        self.process_btn = GlowingButton("开始处理")
        self.process_btn.setMinimumHeight(50)
        self.process_btn.clicked.connect(self.process_video)
        control_layout.addWidget(self.process_btn)
        
        # 创建进度区域
        progress_frame = QFrame()
        progress_layout = QVBoxLayout(progress_frame)
        progress_layout.setContentsMargins(5, 5, 5, 5)
        progress_layout.setSpacing(5)
        
        # 添加圆形进度条
        progress_container = QHBoxLayout()
        
        self.circle_progress = CircularProgressBar()
        self.circle_progress.setValue(0)
        progress_container.addWidget(self.circle_progress, alignment=Qt.AlignCenter)
        
        progress_layout.addLayout(progress_container)
        
        # 添加状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            font-size: 14px;
            color: #03a9f4;
            margin-top: 10px;
        """)
        progress_layout.addWidget(self.status_label)
        
        control_layout.addWidget(progress_frame)
        control_layout.addStretch()
        
        # 添加到主分割器
        self.main_splitter.addWidget(control_frame)
        
    def create_status_bar(self):
        """创建状态栏"""
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #101010;
                color: #bbbbbb;
                border-top: 1px solid #333333;
            }
        """)
        self.statusBar().showMessage("就绪")
        
    def select_video(self):
        """选择视频文件"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "选择视频文件", 
            "", 
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        )
        
        if file_name:
            self.video_path = file_name
            self.video_path_label.setText(os.path.basename(file_name))
            self.statusBar().showMessage(f"已选择视频: {os.path.basename(file_name)}")
            
            # 自动生成输出路径
            if not self.output_path:
                base_name = os.path.basename(file_name)
                name, ext = os.path.splitext(base_name)
                self.output_path = os.path.join(self.output_dir, f"{name}_swapped{ext}")
                self.output_path_label.setText(os.path.basename(self.output_path))
            
            # 加载视频预览
            self.load_video_preview()
    
    def select_face_image(self):
        """选择人脸图片"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "选择人脸图片", 
            "", 
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*.*)"
        )
        
        if file_name:
            self.face_image_path = file_name
            self.face_path_label.setText(os.path.basename(file_name))
            self.statusBar().showMessage(f"已选择人脸图片: {os.path.basename(file_name)}")
            
            # 显示选择的人脸图片
            self.display_face_image()
    
    def select_output(self):
        """选择输出文件"""
        file_name, _ = QFileDialog.getSaveFileName(
            self, 
            "选择输出文件", 
            self.output_path if self.output_path else self.output_dir, 
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        )
        
        if file_name:
            self.output_path = file_name
            self.output_path_label.setText(os.path.basename(file_name))
            self.statusBar().showMessage(f"已设置输出文件: {os.path.basename(file_name)}")
            
    def display_face_image(self):
        """显示选择的人脸图片"""
        if self.face_image_path and os.path.exists(self.face_image_path):
            try:
                # 使用PIL加载图片
                img = Image.open(self.face_image_path)
                img = img.convert("RGB")
                
                # 转换为QImage并显示
                img_qt = QImage(self.face_image_path)
                pixmap = QPixmap.fromImage(img_qt)
                
                # 添加到界面中，实际实现中需要一个显示区域
                # 这里暂时使用状态栏通知用户
                self.statusBar().showMessage(f"已加载人脸图片: {os.path.basename(self.face_image_path)}")
            except Exception as e:
                logger.error(f"加载人脸图片错误: {e}")
                self.statusBar().showMessage(f"加载人脸图片错误: {e}")
    
    def load_video_preview(self):
        """加载视频预览"""
        if self.video_path and os.path.exists(self.video_path):
            if self.video_thread:
                self.video_thread.stop()
                
            self.video_thread = VideoThread(self.video_path)
            self.video_thread.frame_ready.connect(self.update_frame)
            self.video_thread.duration_updated.connect(self.update_duration)
            self.video_thread.position_updated.connect(self.update_position)
            self.video_thread.start()
            
            # 更新状态
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        
    def toggle_video(self):
        """切换视频播放状态"""
        if self.video_thread:
            self.video_thread.toggle_pause()
            
            if self.video_thread.paused:
                self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            else:
                self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        elif self.video_path:
            self.load_video_preview()
        else:
            self.statusBar().showMessage("请先选择视频文件")
    
    def stop_video(self):
        """停止视频播放"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.video_label.clear()
            self.current_time.setText("00:00")
            self.time_slider.setValue(0)
    
    def update_frame(self, image):
        """更新视频帧"""
        self.video_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def update_duration(self, duration):
        """更新视频总时长"""
        self.total_time.setText(self.format_time(duration))
        
    def update_position(self, position):
        """更新当前播放位置"""
        self.current_time.setText(self.format_time(position))
        
        # 避免信号循环
        self.time_slider.blockSignals(True)
        if self.video_thread and self.video_thread.fps > 0:
            duration = self.video_thread.total_frames / self.video_thread.fps
            if duration > 0:
                self.time_slider.setValue(int(position / duration * 100))
        self.time_slider.blockSignals(False)
    
    def seek_video(self, value):
        """视频跳转"""
        if self.video_thread and self.video_thread.fps > 0:
            # 计算秒数
            duration = self.video_thread.total_frames / self.video_thread.fps
            position = duration * value / 100
            self.video_thread.seek(position)
    
    def format_time(self, seconds):
        """格式化时间显示"""
        minutes = int(seconds / 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
        
    def process_video(self):
        """处理视频"""
        if not self.video_path or not self.face_image_path:
            self.status_label.setText("请先选择视频和人脸图片")
            self.statusBar().showMessage("请先选择视频和人脸图片")
            return
            
        if not self.output_path:
            # 自动生成输出路径
            base_name = os.path.basename(self.video_path)
            name, ext = os.path.splitext(base_name)
            self.output_path = os.path.join(self.output_dir, f"{name}_swapped{ext}")
            self.output_path_label.setText(os.path.basename(self.output_path))
            
        # 更新状态
        self.status_label.setText("正在准备处理...")
        self.statusBar().showMessage("正在准备处理视频，请稍候...")
        self.circle_progress.setValue(0)
        
        # 检查目录是否存在
        output_dir = os.path.dirname(self.output_path)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                self.status_label.setText(f"创建输出目录失败: {e}")
                self.statusBar().showMessage(f"创建输出目录失败: {e}")
                return
        
        # 获取处理选项