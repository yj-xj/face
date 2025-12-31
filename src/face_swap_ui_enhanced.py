import sys
import os
import cv2
import time
import numpy as np
from enum import Enum
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QLabel, QFileDialog,
                            QProgressBar, QFrame, QSlider, QStyle, QComboBox,
                            QScrollArea, QGridLayout, QGroupBox, QCheckBox,
                            QSplitter, QStackedWidget, QDialog, QSpacerItem,
                            QSizePolicy, QMenu, QToolButton, QAction, QLineEdit,
                            QListWidget, QMessageBox, QButtonGroup, QRadioButton,
                            QListWidgetItem, QGraphicsOpacityEffect)
from PyQt5.QtCore import (Qt, QThread, pyqtSignal, QTimer, QSize, QUrl,
                          QPropertyAnimation, QEasingCurve, QRect, QPoint,
                          QParallelAnimationGroup, QSequentialAnimationGroup)
from PyQt5.QtGui import (QImage, QPixmap, QPalette, QColor, QFont,
                        QCursor, QIcon, QRadialGradient, QLinearGradient,
                        QPainter, QPen, QBrush, QFontDatabase)
from PIL import Image
import logging
import qdarktheme
from PyQt5.QtMultimedia import QMediaPlayer

# 检查是否需要PyQt5 Multimedia插件
try:
    from PyQt5.QtMultimediaWidgets import QVideoWidget
except ImportError:
    QMessageBox.critical(None, "错误", "未找到PyQt5 Multimedia插件，请确保安装了PyQt5.QtMultimedia")
    sys.exit(1)

# 导入原始的人脸替换功能
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from face_swap import FaceSwapApp as OriginalFaceSwapApp

# 尝试导入dlib（如果可用）
try:
    import dlib
    HAS_DLIB = True
except ImportError:
    HAS_DLIB = False
    print("警告: dlib模块未安装，某些功能可能不可用")

# 定义应用模式枚举
class AppMode(Enum):
    VIDEO_MODE = "video"
    CAMERA_MODE = "camera"

class VideoProcessingThread(QThread):
    """视频处理线程，防止UI卡顿（优化版）"""
    progress_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, face_swap_app, video_path, face_image_path, output_path):
        super().__init__()
        self.face_swap_app = face_swap_app
        self.video_path = video_path
        self.face_image_path = face_image_path
        self.output_path = output_path
        self.last_progress_update = 0  # 上次更新进度的时间
        self.progress_update_interval = 0.5  # 进度更新间隔（秒）

    def run(self):
        try:
            # 设置处理参数
            self.face_swap_app.video_path = self.video_path
            self.face_swap_app.face_images = [self.face_image_path]
            self.face_swap_app.selected_face_index = 0
            self.face_swap_app.output_path = self.output_path

            # 保存原始更新方法
            original_update_progress = self.face_swap_app.update_progress
            original_update_status = self.face_swap_app.update_status

            # 创建节流的更新方法，减少信号发射频率
            def new_update_progress(value, text=None):
                current_time = time.time()
                # 只在间隔超过阈值时更新进度
                if current_time - self.last_progress_update >= self.progress_update_interval or value >= 100:
                    self.progress_signal.emit(int(value))
                    self.last_progress_update = current_time
                # 调用原始方法记录日志等操作（但不发送UI信号）
                original_update_progress(value, text)

            def new_update_status(text):
                # 减少状态更新频率
                self.status_signal.emit(text)
                # 调用原始方法记录日志等操作
                original_update_status(text)

            # 覆盖FaceSwapApp的方法
            self.face_swap_app.update_progress = new_update_progress
            self.face_swap_app.update_status = new_update_status

            # 调用原始处理方法
            result = self.face_swap_app.process_video()

            # 处理完成后发送信号
            if result and isinstance(result, str) and os.path.exists(result):
                self.finished_signal.emit(result)
            else:
                # 如果返回值为True但不是文件路径
                if result is True and os.path.exists(self.output_path):
                    self.finished_signal.emit(self.output_path)
                else:
                    self.error_signal.emit("处理完成，但无法生成有效的输出视频。")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(str(e))

class CameraProcessingThread(QThread):
    """摄像头实时处理线程，用于实时人脸检测和替换"""
    frame_ready = pyqtSignal(np.ndarray)  # 发送处理后的帧
    status_signal = pyqtSignal(str)  # 发送状态信息
    error_signal = pyqtSignal(str)  # 发送错误信息

    def __init__(self, face_swap_app, target_face_path):
        super().__init__()
        self.face_swap_app = face_swap_app
        self.target_face_path = target_face_path
        self.running = False
        self.camera = None
        self.camera_index = 0  # 默认摄像头索引
        self.processing_enabled = True  # 是否启用人脸替换

    def set_camera_index(self, index):
        """设置摄像头索引"""
        self.camera_index = index

    def set_target_face(self, face_path):
        """设置目标人脸"""
        self.target_face_path = face_path

    def set_processing_enabled(self, enabled):
        """设置是否启用人脸替换"""
        self.processing_enabled = enabled

    def start_camera(self):
        """启动摄像头"""
        self.running = True
        self.start()

    def stop_camera(self):
        """停止摄���头"""
        self.running = False
        self.wait(3000)  # 等待线程结束，最多3秒

    def run(self):
        """摄像头处理主循环"""
        try:
            # 打开摄像头
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                self.error_signal.emit(f"无法打开摄像头 (索引: {self.camera_index})")
                return

            # 获取摄像头参数
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.camera.get(cv2.CAP_PROP_FPS)

            self.status_signal.emit(f"摄像头已启动: {width}x{height} @ {fps}fps")

            # 读取目标人脸
            target_face = None

            if self.target_face_path and os.path.exists(self.target_face_path):
                target_face = cv2.imread(self.target_face_path)
                if target_face is not None:
                    self.status_signal.emit("目标人脸已加载 (使用Inswapper模型)")
                else:
                    self.error_signal.emit(f"无法读取目标人脸: {self.target_face_path}")
            else:
                self.status_signal.emit("未选择目标人脸，仅显示摄像头画面")

            # 主循环 - 优化版
            frame_count = 0
            skip_frames = 1  # 跳帧处理以提高性能

            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    self.error_signal.emit("无法从摄像头读取帧")
                    break

                # 跳帧处理以提高帧率
                frame_count += 1
                if frame_count % skip_frames != 0:
                    # 跳过的帧直接显示原始画面
                    self.frame_ready.emit(frame)
                    continue

                # 如果启用了处理且加载了目标人脸
                if self.processing_enabled and target_face is not None:
                    try:
                        # 使用inswapper模型进行实时换脸
                        if self.face_swap_app.inswapper is not None and self.face_swap_app.face_analyser is not None:
                            # 降低分辨率以加快处理速度
                            h, w = frame.shape[:2]
                            if w > 640:  # 如果宽度超过640，缩小处理
                                scale = 640 / w
                                small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                                processed_small = self.face_swap_app.insightface_face_swap(small_frame, target_face)
                                if processed_small is not None:
                                    processed_frame = cv2.resize(processed_small, (w, h))
                                    self.frame_ready.emit(processed_frame)
                                else:
                                    self.frame_ready.emit(frame)
                            else:
                                # 小尺寸视频直接处理
                                processed_frame = self.face_swap_app.insightface_face_swap(frame, target_face)
                                if processed_frame is not None:
                                    self.frame_ready.emit(processed_frame)
                                else:
                                    self.frame_ready.emit(frame)
                        else:
                            # 如果inswapper不可用，显示原始帧
                            self.frame_ready.emit(frame)

                    except Exception as e:
                        # 处理出错时显示原始帧
                        self.frame_ready.emit(frame)
                        import traceback
                        traceback.print_exc()
                else:
                    # 直接发送原始帧
                    self.frame_ready.emit(frame)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(str(e))
        finally:
            # 释放摄像头
            if self.camera is not None and self.camera.isOpened():
                self.camera.release()
                self.status_signal.emit("摄像头已关闭")


class CircularProgressBar(QWidget):
    """自定义圆形进度条"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(150, 150)
        self.value = 0
        self.max_value = 100
        self.setStyleSheet("background-color: transparent;")
        
    def setValue(self, value):
        self.value = value
        self.update()
        
    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter, QColor, QPen
        from PyQt5.QtCore import QRectF, Qt
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 计算圆形区域
        rect = QRectF(10, 10, self.width() - 20, self.height() - 20)
        
        # 绘制背景圆环
        painter.setPen(QPen(QColor(50, 50, 50), 10))
        painter.drawArc(rect, 0, 360 * 16)  # 角度乘以16是Qt的要求
        
        # 绘制进度圆环
        painter.setPen(QPen(QColor(66, 135, 245), 10))
        span_angle = int(-self.value * 360 / self.max_value * 16)
        painter.drawArc(rect, 90 * 16, span_angle)
        
        # 绘制文本
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont('Arial', 20, QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, f"{int(self.value)}%")

class GlowingButton(QPushButton):
    """自定义带发光效果的按钮"""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: #4287f5;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5a9cff;
                border: 2px solid #85b7ff;
            }
            QPushButton:pressed {
                background-color: #3a78d1;
            }
        """)
        self.setMinimumHeight(40)

class EnhancedFaceSwapUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # ========== 模式管理 ==========
        self.current_mode = AppMode.VIDEO_MODE  # 默认视频模式
        self.mode_animation = None  # 模式切换动画

        # 初始化原始FaceSwapApp的实例
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

        # 摄像头相关
        self.camera_thread = None
        self.camera_active = False
        self.camera_processing_enabled = True

        # 播放速度控制
        self.playback_speed_factor = 1.0
        self.speed_options = {
            "0.25x": 0.25, "0.5x": 0.5, "0.75x": 0.75,
            "1.0x": 1.0, "1.25x": 1.25, "1.5x": 1.5, "2.0x": 2.0
        }

        # 设置窗口属性
        if self.original_app.root is not None:
            self.original_app.root.title("人脸替换应用 - InsightFace版")
            self.original_app.root.geometry("1200x800")
            # 设置应用程序图标和全局字体
            self.original_app.set_app_appearance()

        # 设置窗口属性
        self.setWindowTitle("人脸替换应用 - 增强版 (支持视频/摄像头)")
        self.resize(1600, 900)  # 更大的窗口尺寸

        # 启用拖放功能
        self.setAcceptDrops(True)

        # 设置应用程序图标
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            app_icon_path = os.path.join(base_dir, "resources", "app_icon.png")
            if os.path.exists(app_icon_path):
                self.setWindowIcon(QIcon(app_icon_path))
        except Exception as e:
            print(f"加载应用图标失败: {e}")

        # 初始化UI组件
        self.initUI()

        # 加载文件列表
        self.loadFaceImages()
        self.loadVideoFiles()

        # 初始化视频播放器
        self.initMediaPlayer()
        
    def initUI(self):
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ========== 顶部标题栏（包含模式切换） ==========
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel("人脸替换应用")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #4287f5;
        """)
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # 创建模式切换按钮组
        mode_widget = QWidget()
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(5)

        # 视频模式按钮
        self.video_mode_btn = QPushButton("[视频] 视频模式")
        self.video_mode_btn.setCheckable(True)
        self.video_mode_btn.setChecked(True)
        self.video_mode_btn.setStyleSheet("""
            QPushButton {
                background-color: #4287f5;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:checked {
                background-color: #5a9cff;
                border: 2px solid #85b7ff;
            }
            QPushButton:hover {
                background-color: #5a9cff;
            }
        """)
        self.video_mode_btn.clicked.connect(lambda: self.switchMode(AppMode.VIDEO_MODE))
        mode_layout.addWidget(self.video_mode_btn)

        # 摄像头模式按钮
        self.camera_mode_btn = QPushButton("[摄像头] 摄像头模式")
        self.camera_mode_btn.setCheckable(True)
        self.camera_mode_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #444444;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:checked {
                background-color: #4287f5;
                border: 2px solid #85b7ff;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
        """)
        self.camera_mode_btn.clicked.connect(lambda: self.switchMode(AppMode.CAMERA_MODE))
        mode_layout.addWidget(self.camera_mode_btn)

        header_layout.addWidget(mode_widget)
        main_layout.addLayout(header_layout)
        
        # 创建分割器，将界面分为左右两部分
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #444444;
            }
        """)
        main_layout.addWidget(splitter, 1)  # 1表示拉伸因子
        
        # 创建左侧视频播放部分
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 视频播放区域
        video_frame = QFrame()
        video_frame.setStyleSheet("""
            background-color: #111111; 
            border: 2px solid #444444;
            border-radius: 5px;
        """)
        video_frame.setMinimumSize(800, 600)
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(5, 5, 5, 5)
        
        # 创建视频显示标签
        self.cv_video_label = QLabel()
        self.cv_video_label.setAlignment(Qt.AlignCenter)
        self.cv_video_label.setStyleSheet("""
            background-color: #000000;
            border: 1px solid #333333;
            border-radius: 3px;
        """)
        self.cv_video_label.setText("<font color='#888888'>准备播放视频</font>")
        self.cv_video_label.setFont(QFont('Arial', 24))
        video_layout.addWidget(self.cv_video_label)
        
        # 添加到左侧布局
        left_layout.addWidget(video_frame)
        
        # 保留QVideoWidget但隐藏它，以兼容旧代码
        self.video_widget = QVideoWidget()
        self.video_widget.hide()

        # ========== 创建独立的控制栏系统 ==========
        # 使用QStackedWidget管理视频和摄像头的控制栏
        self.control_bar_stack = QStackedWidget()
        left_layout.addWidget(self.control_bar_stack)

        # ========== 视频模式控制栏 ==========
        self.video_control_bar = QWidget()
        video_control_layout = QHBoxLayout(self.video_control_bar)
        video_control_layout.setContentsMargins(5, 5, 5, 5)
        video_control_layout.setSpacing(8)

        # 添加打开视频按钮（带美化动效）
        open_video_btn = QPushButton("打开视频")
        open_video_btn.setIcon(QIcon.fromTheme("document-open"))
        open_video_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4a90e2, stop:1 #357abd);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5a9fff, stop:1 #4682b4);
                box-shadow: 0 4px 8px rgba(74, 144, 226, 0.3);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #357abd, stop:1 #286090);
            }
        """)
        open_video_btn.clicked.connect(self.openVideo)
        video_control_layout.addWidget(open_video_btn)

        # 播放/暂停按钮
        self.play_pause_button = QPushButton("播放")
        self.play_pause_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_pause_button.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #444444;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
                border: 1px solid #666666;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
            }
            QPushButton:pressed {
                background-color: #222222;
            }
        """)
        self.play_pause_button.clicked.connect(self.togglePlayback)
        video_control_layout.addWidget(self.play_pause_button)

        # 停止按钮
        self.stop_button = QPushButton()
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.setToolTip("停止")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                border: 1px solid #444444;
                border-radius: 6px;
                padding: 8px;
                min-width: 36px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
                border: 1px solid #666666;
            }
        """)
        self.stop_button.clicked.connect(self.stopPlayback)
        video_control_layout.addWidget(self.stop_button)

        # 播放速度控制
        self.speed_combo_box = QComboBox()
        self.speed_combo_box.addItems(self.speed_options.keys())
        self.speed_combo_box.setCurrentText("1.0x")
        self.speed_combo_box.setToolTip("调整播放速度")
        self.speed_combo_box.setStyleSheet("""
            QComboBox {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #444444;
                border-radius: 6px;
                padding: 5px 10px;
                min-width: 80px;
            }
            QComboBox:hover {
                border: 1px solid #666666;
                background-color: #333333;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #2a2a2a;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                color: white;
                selection-background-color: #4287f5;
                border: 1px solid #444444;
            }
        """)
        self.speed_combo_box.currentTextChanged.connect(self.changePlaybackSpeed)
        video_control_layout.addWidget(self.speed_combo_box)

        # 进度条
        self.position_slider = QProgressBar()
        self.position_slider.setTextVisible(False)
        self.position_slider.setRange(0, 100)
        self.position_slider.setValue(0)
        self.position_slider.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: #1a1a1a;
                height: 8px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4287f5, stop:1 #5a9fff);
                border-radius: 4px;
            }
        """)

        # 时间标签
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 12px;
                padding: 0 8px;
                background-color: #1a1a1a;
                border-radius: 4px;
                padding: 4px 10px;
            }
        """)

        video_control_layout.addWidget(self.position_slider, 1)
        video_control_layout.addWidget(self.time_label)

        # 添加视频控制栏到堆叠窗口
        self.control_bar_stack.addWidget(self.video_control_bar)

        # ========== 摄像头模式控制栏（简化版） ==========
        self.camera_control_bar = QWidget()
        camera_control_layout = QHBoxLayout(self.camera_control_bar)
        camera_control_layout.setContentsMargins(5, 5, 5, 5)
        camera_control_layout.setSpacing(8)

        # 摄像头状态指示器
        self.camera_status_indicator = QLabel()
        self.camera_status_indicator.setFixedSize(12, 12)
        self.camera_status_indicator.setStyleSheet("""
            QLabel {
                background-color: #666666;
                border-radius: 6px;
                border: 2px solid #888888;
            }
        """)
        camera_control_layout.addWidget(self.camera_status_indicator)

        # 摄像头状态文本
        self.camera_bar_status_label = QLabel("摄像头未启动")
        self.camera_bar_status_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 13px;
                padding: 4px 10px;
                background-color: #1a1a1a;
                border-radius: 4px;
            }
        """)
        camera_control_layout.addWidget(self.camera_bar_status_label)

        camera_control_layout.addStretch()

        # 分辨率显示
        self.camera_resolution_label = QLabel()
        self.camera_resolution_label.setStyleSheet("""
            QLabel {
                color: #666666;
                font-size: 12px;
            }
        """)
        camera_control_layout.addWidget(self.camera_resolution_label)

        # 添加摄像头控制栏到堆叠窗口
        self.control_bar_stack.addWidget(self.camera_control_bar)

        # 默认显示视频控制栏
        self.control_bar_stack.setCurrentWidget(self.video_control_bar)

        # ========== 创建右侧控制部分（使用QStackedWidget分离两种模式） ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 创建堆叠窗口部件用于模式切换
        self.control_stack = QStackedWidget()
        right_layout.addWidget(self.control_stack)

        # ========== 创建视频模式控制面板 ==========
        self.video_control_panel = QWidget()
        video_panel_layout = QVBoxLayout(self.video_control_panel)
        video_panel_layout.setContentsMargins(0, 0, 0, 0)

        # 人脸图片选择（视频模式）
        video_face_group = QGroupBox("选择人脸图片")
        video_face_layout = QVBoxLayout(video_face_group)

        video_face_label = QLabel("选择要替换的人脸:")
        video_face_label.setStyleSheet("font-weight: bold;")
        video_face_layout.addWidget(video_face_label)

        self.video_face_list = QListWidget()
        self.video_face_list.setFixedHeight(150)
        self.video_face_list.setStyleSheet("""
            QListWidget {
                background-color: #2a2a2a;
                border: 1px solid #444444;
                border-radius: 5px;
            }
            QListWidget::item {
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #4287f5;
            }
        """)
        self.video_face_list.setViewMode(QListWidget.IconMode)
        self.video_face_list.setIconSize(QSize(100, 100))
        self.video_face_list.setResizeMode(QListWidget.Adjust)
        self.video_face_list.setSpacing(8)
        self.video_face_list.itemClicked.connect(self.selectFaceImage)
        video_face_layout.addWidget(self.video_face_list)

        video_panel_layout.addWidget(video_face_group)

        # 视频文件选择
        video_input_group = QGroupBox("视频处理")
        video_input_layout = QVBoxLayout(video_input_group)

        video_label = QLabel("选择视频文件:")
        video_label.setStyleSheet("font-weight: bold;")
        video_input_layout.addWidget(video_label)

        self.video_list = QListWidget()
        self.video_list.setFixedHeight(200)
        self.video_list.setStyleSheet("""
            QListWidget {
                background-color: #2a2a2a;
                border: 1px solid #444444;
                border-radius: 5px;
            }
            QListWidget::item {
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #4287f5;
            }
        """)
        self.video_list.setIconSize(QSize(200, 150))
        self.video_list.itemClicked.connect(self.selectVideoFile)
        video_input_layout.addWidget(self.video_list)

        # 输出文件选择
        output_layout = QHBoxLayout()
        output_label = QLabel("输出路径:")
        output_label.setStyleSheet("font-weight: bold;")
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("选择或输入输出视频路径...")
        self.output_path_edit.setStyleSheet("""
            QLineEdit {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #444444;
                border-radius: 3px;
                padding: 5px;
            }
        """)

        browse_output_btn = QPushButton("浏览...")
        browse_output_btn.clicked.connect(self.browseOutputPath)

        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_path_edit, 1)
        output_layout.addWidget(browse_output_btn)

        video_input_layout.addLayout(output_layout)
        video_panel_layout.addWidget(video_input_group)

        # 添加高级选项部分
        advanced_group = QGroupBox("高级选项")
        advanced_layout = QGridLayout(advanced_group)

        # 平滑度滑块
        smooth_label = QLabel("面部混合平滑度:")
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(0, 100)
        self.smooth_slider.setValue(50)
        self.smooth_slider.setStyleSheet("""
            QSlider {
                height: 20px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #444444;
                height: 8px;
                background: #2a2a2a;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4287f5;
                border: 1px solid #4287f5;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)

        # 颜色校正选项
        self.color_correction_check = QCheckBox("启用颜色校正")
        self.color_correction_check.setChecked(True)

        # 多尺度检测选项
        self.multi_scale_check = QCheckBox("启用多尺度人脸检测 (更精确但更慢)")
        self.multi_scale_check.setChecked(True)

        # 人脸检测器选择
        detector_label = QLabel("人脸检测器:")
        self.detector_group = QButtonGroup(self)
        self.dlib_radio = QRadioButton("Dlib")
        self.opencv_radio = QRadioButton("OpenCV")
        self.dlib_radio.setChecked(True)
        self.detector_group.addButton(self.dlib_radio)
        self.detector_group.addButton(self.opencv_radio)

        detector_layout = QHBoxLayout()
        detector_layout.addWidget(self.dlib_radio)
        detector_layout.addWidget(self.opencv_radio)

        # 人脸替换方法选择
        swapper_label = QLabel("人脸替换方法:")
        self.swapper_group = QButtonGroup(self)
        self.traditional_radio = QRadioButton("传统方法")
        self.inswapper_radio = QRadioButton("Inswapper")

        # 只有当inswapper模型可用时才启用此选项
        self.inswapper_radio.setEnabled(self.original_app.inswapper is not None and self.original_app.face_analyser is not None)

        if self.original_app.inswapper is not None and self.original_app.face_analyser is not None:
            self.inswapper_radio.setChecked(True)
        else:
            self.traditional_radio.setChecked(True)

        self.swapper_group.addButton(self.traditional_radio)
        self.swapper_group.addButton(self.inswapper_radio)

        swapper_layout = QHBoxLayout()
        swapper_layout.addWidget(self.traditional_radio)
        swapper_layout.addWidget(self.inswapper_radio)

        # 布局网格
        advanced_layout.addWidget(smooth_label, 0, 0)
        advanced_layout.addWidget(self.smooth_slider, 0, 1)
        advanced_layout.addWidget(self.color_correction_check, 1, 0, 1, 2)
        advanced_layout.addWidget(self.multi_scale_check, 2, 0, 1, 2)
        advanced_layout.addWidget(detector_label, 3, 0)
        advanced_layout.addLayout(detector_layout, 3, 1)
        advanced_layout.addWidget(swapper_label, 4, 0)
        advanced_layout.addLayout(swapper_layout, 4, 1)

        video_panel_layout.addWidget(advanced_group)

        # 视频处理控制
        video_process_group = QGroupBox("处理控制")
        video_process_layout = QVBoxLayout(video_process_group)

        # 进度显示区域
        progress_layout = QHBoxLayout()

        self.circular_progress = CircularProgressBar()
        self.circular_progress.setValue(0)

        status_layout = QVBoxLayout()
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("""
            font-size: 14px;
            color: white;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)

        self.process_button = GlowingButton("开始处理")
        self.process_button.clicked.connect(self.startProcessing)

        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.process_button)
        status_layout.addStretch()

        progress_layout.addWidget(self.circular_progress)
        progress_layout.addLayout(status_layout)

        video_process_layout.addLayout(progress_layout)
        video_panel_layout.addWidget(video_process_group)

        # 添加视频面板到堆叠窗口
        self.control_stack.addWidget(self.video_control_panel)

        # ========== 创建摄像头模式控制面板 ==========
        self.camera_control_panel = QWidget()
        camera_panel_layout = QVBoxLayout(self.camera_control_panel)
        camera_panel_layout.setContentsMargins(0, 0, 0, 0)

        # 摄像头人脸选择
        camera_input_group = QGroupBox("摄像头实时换脸")
        camera_input_layout = QVBoxLayout(camera_input_group)

        face_label = QLabel("选择要替换的人脸:")
        face_label.setStyleSheet("font-weight: bold;")
        camera_input_layout.addWidget(face_label)

        self.face_list = QListWidget()
        self.face_list.setFixedHeight(250)
        self.face_list.setStyleSheet("""
            QListWidget {
                background-color: #2a2a2a;
                border: 1px solid #444444;
                border-radius: 5px;
            }
            QListWidget::item {
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #4287f5;
            }
        """)
        self.face_list.setViewMode(QListWidget.IconMode)
        self.face_list.setIconSize(QSize(130, 130))
        self.face_list.setResizeMode(QListWidget.Adjust)
        self.face_list.setSpacing(10)
        self.face_list.itemClicked.connect(self.selectFaceImage)
        camera_input_layout.addWidget(self.face_list)

        camera_panel_layout.addWidget(camera_input_group)

        # 摄像头控制
        camera_control_group = QGroupBox("摄像头控制")
        camera_control_group_layout = QVBoxLayout(camera_control_group)

        # 摄像头按钮行
        camera_btn_layout = QHBoxLayout()

        self.camera_toggle_btn = QPushButton("开启摄像头")
        self.camera_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        self.camera_toggle_btn.clicked.connect(self.toggleCamera)
        camera_btn_layout.addWidget(self.camera_toggle_btn)

        # 拍照按钮
        self.snapshot_btn = QPushButton("[拍照] 保存快照")
        self.snapshot_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
            QPushButton:disabled {
                background-color: #444444;
                color: #888888;
            }
        """)
        self.snapshot_btn.clicked.connect(self.takeSnapshot)
        self.snapshot_btn.setEnabled(False)
        camera_btn_layout.addWidget(self.snapshot_btn)

        camera_control_group_layout.addLayout(camera_btn_layout)

        # 摄像头处理开关
        self.camera_processing_check = QCheckBox("启用实时人脸替换 (Inswapper模型)")
        self.camera_processing_check.setChecked(True)
        self.camera_processing_check.setStyleSheet("""
            QCheckBox {
                color: white;
                font-size: 13px;
                padding: 5px;
            }
        """)
        self.camera_processing_check.setEnabled(True)
        camera_control_group_layout.addWidget(self.camera_processing_check)

        # 摄像头状态信息
        self.camera_status_label = QLabel("摄像头未启动")
        self.camera_status_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 12px;
                padding: 5px;
                border: 1px solid #444444;
                border-radius: 3px;
                background-color: #1a1a1a;
            }
        """)
        self.camera_status_label.setAlignment(Qt.AlignCenter)
        camera_control_group_layout.addWidget(self.camera_status_label)

        camera_panel_layout.addWidget(camera_control_group)

        # 添加使用提示
        tip_group = QGroupBox("使用提示")
        tip_layout = QVBoxLayout(tip_group)
        tip_text = QLabel(
            "1. 从上方选择要替换的人脸图片\n"
            "2. 点击 '开启摄像头' 启动摄像头\n"
            "3. 实时查看换脸效果\n"
            "4. 点击 '保存快照' 保存当前画面\n"
            "5. 取消勾选可仅显示摄像头画面"
        )
        tip_text.setStyleSheet("""
            QLabel {
                color: #aaaaaa;
                font-size: 12px;
                padding: 5px;
            }
        """)
        tip_text.setWordWrap(True)
        tip_layout.addWidget(tip_text)
        camera_panel_layout.addWidget(tip_group)

        camera_panel_layout.addStretch()

        # 添加摄像头面板到堆叠窗口
        self.control_stack.addWidget(self.camera_control_panel)

        # 默认显示视频面板
        self.control_stack.setCurrentWidget(self.video_control_panel)

        # 添加到分割器
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        
        # 设置初始分割比例（左:右 = 3:2）
        splitter.setSizes([900, 600])
        
        # 添加状态栏
        self.statusBar().showMessage("就绪")
        
    def initMediaPlayer(self):
        """初始化媒体播放器"""
        try:
            # 创建媒体播放器并设置视频输出
            self.media_player = QMediaPlayer(self)
            self.media_player.setVideoOutput(self.video_widget)
            
            # 添加错误处理
            self.media_player.error.connect(self.handleMediaError)
            
            # 连接信号
            self.media_player.stateChanged.connect(self.mediaStateChanged)
            self.media_player.positionChanged.connect(self.positionChanged)
            self.media_player.durationChanged.connect(self.durationChanged)
            
            print("媒体播放器初始化成功")
        except Exception as e:
            print(f"媒体播放器初始化失败: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "警告", f"初始化媒体播放器失败: {e}")
    
    def handleMediaError(self, error):
        """处理媒体播放器错误"""
        error_messages = {
            QMediaPlayer.NoError: "无错误",
            QMediaPlayer.ResourceError: "资源错误：无法打开媒体资源",
            QMediaPlayer.FormatError: "格式错误：不支持的媒体格式",
            QMediaPlayer.NetworkError: "网络错误：网络访问失败",
            QMediaPlayer.AccessDeniedError: "访问被拒绝：无权访问内容",
            QMediaPlayer.ServiceMissingError: "服务缺失：没有可用的媒体服务"
        }
        
        error_msg = error_messages.get(error, f"未知错误：{error}")
        QMessageBox.warning(self, "媒体播放错误", error_msg)
        self.statusBar().showMessage(f"播放错误: {error_msg}")
        
        # 尝试使用备用方式播放
        if error != QMediaPlayer.NoError:
            self.tryAlternativePlay()
    
    def togglePlayback(self):
        """切换播放/暂停状态"""
        if hasattr(self, 'cv_play_timer') and hasattr(self, 'cv_cap') and self.cv_cap is not None and self.cv_cap.isOpened():
            if self.cv_play_timer.isActive():
                # 暂停OpenCV播放
                self.cv_play_timer.stop()
                self.play_pause_button.setText("播放")
            else:
                # 继续OpenCV播放
                if self.cv_fps <= 0:
                    self.cv_fps = 30  # 默认30fps
                delay = int(1000 / (self.cv_fps * self.playback_speed_factor))
                if delay <= 0: # 防止延迟为0或负数
                    delay = 1 # 设置一个最小延迟
                self.cv_play_timer.start(delay)
                self.play_pause_button.setText("暂停")
        # 否则使用QMediaPlayer
        elif hasattr(self, 'media_player'):
            if self.media_player.state() == QMediaPlayer.PlayingState:
                self.media_player.pause()
            else:
                self.media_player.play()
        else:
            # 如果有视频路径但没有播放器，尝试初始化播放
            if hasattr(self, 'current_video_path') and os.path.exists(self.current_video_path):
                self.playWithOpenCV()
    
    def tryAlternativePlay(self):
        """尝试使用替代方式播放视频"""
        if hasattr(self, 'current_video_path') and os.path.exists(self.current_video_path):
            try:
                # 尝试使用OpenCV播放
                answer = QMessageBox.question(self, "播放选项", 
                                            "内置播放器无法播放此视频。\n是否使用OpenCV播放器？\n「是」使用OpenCV播放器\n「否」使用系统默认播放器", 
                                            QMessageBox.Yes | QMessageBox.No)
                
                if answer == QMessageBox.Yes:
                    self.playWithOpenCV()
                else:
                    # 使用系统默认播放器打开
                    import subprocess
                    import platform
                    
                    system = platform.system()
                    if system == 'Windows':
                        os.startfile(self.current_video_path)
                    elif system == 'Darwin':  # macOS
                        subprocess.call(('open', self.current_video_path))
                    else:  # Linux
                        subprocess.call(('xdg-open', self.current_video_path))
                        
            except Exception as e:
                QMessageBox.warning(self, "错误", f"无法播放视频: {e}")
                import traceback
                traceback.print_exc()
            
    def playWithOpenCV(self):
        """使用OpenCV播放视频"""
        try:
            # 检查是否已经有正在运行的播放线程
            if hasattr(self, 'cv_play_timer') and self.cv_play_timer.isActive():
                self.cv_play_timer.stop()
            
            # 释放旧的视频捕获
            if hasattr(self, 'cv_cap') and self.cv_cap is not None and self.cv_cap.isOpened():
                print("释放旧的视频捕获")
                self.cv_cap.release()
                self.cv_cap = None
            
            print(f"尝试使用OpenCV打开视频: {self.current_video_path}")
            # 显示加载提示
            self.cv_video_label.setText("<font color='#AAAAAA'>正在加载视频...</font>")
            self.cv_video_label.repaint()  # 立即刷新UI
            
            # 打开视频
            self.cv_cap = cv2.VideoCapture(self.current_video_path)
            if not self.cv_cap.isOpened():
                error_msg = f"OpenCV无法打开视频文件: {self.current_video_path}"
                print(error_msg)
                self.cv_video_label.setText(f"<font color='#FF5555'>视频加载失败</font>")
                raise Exception(error_msg)
            
            # 获取视频基本信息
            width = int(self.cv_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cv_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.cv_fps = self.cv_cap.get(cv2.CAP_PROP_FPS)
            self.cv_frame_count = int(self.cv_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.cv_current_frame = 0
            
            # 清除文本，准备显示视频
            self.cv_video_label.clear()
            
            # 显示视频信息
            video_info = f"视频信息: {width}x{height}, {self.cv_fps:.2f}fps, {self.cv_frame_count}帧"
            print(video_info)
            self.statusBar().showMessage(video_info)
            
            # 计算延迟（毫秒）- 限制最大和最小帧率
            if self.cv_fps <= 0 or self.cv_fps > 120:
                self.cv_fps = 30  # 默认30fps
            delay = int(1000 / (self.cv_fps * self.playback_speed_factor))
            if delay <= 0: # 防止延迟为0或负数
                delay = 1 # 设置一个最小延迟
            
            # 读取第一帧显示
            ret, frame = self.cv_cap.read()
            if ret:
                print("成功读取第一帧，准备显示")
                self.showFrame(frame)
            else:
                self.cv_video_label.setText("<font color='#FF5555'>无法读取视频帧</font>")
                print("无法读取第一帧")
                return
            
            # 创建新的定时器
            if hasattr(self, 'cv_play_timer') and self.cv_play_timer is not None:
                if self.cv_play_timer.isActive():
                    self.cv_play_timer.stop()
            else:
                self.cv_play_timer = QTimer(self)
            
            self.cv_play_timer.timeout.connect(self.showNextFrame)
            print(f"开始定时播放，间隔: {delay}ms, 速度因子: {self.playback_speed_factor}x")
            self.cv_play_timer.start(delay)
            
            # 更新UI状态
            self.play_pause_button.setText("暂停")
            self.statusBar().showMessage(f"正在使用OpenCV播放: {os.path.basename(self.current_video_path)}")
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"OpenCV播放失败: {e}")
            import traceback
            traceback.print_exc()
    
    def showFrame(self, frame):
        """显示一帧视频（优化版，减少卡顿）"""
        try:
            if frame is None:
                return

            # 转换颜色空间
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 获取标签大小
            label_width = self.cv_video_label.width()
            label_height = self.cv_video_label.height()

            # 如果标签尚未正确初始化，使用父容器大小
            if label_width < 10 or label_height < 10:
                parent = self.cv_video_label.parentWidget()
                if parent:
                    label_width = parent.width() - 10
                    label_height = parent.height() - 10
                else:
                    label_width = 800
                    label_height = 600

            # 转换为QImage
            h, w, c = frame_rgb.shape

            # 使用FastTransformation提升性能
            q_img = QImage(frame_rgb.data, w, h, w * c, QImage.Format_RGB888)

            # 创建QPixmap并调整大小以适应视频标签
            pixmap = QPixmap.fromImage(q_img)

            # 计算缩放比例，保持视频的原始宽高比
            scale_w = label_width / w
            scale_h = label_height / h
            scale = min(scale_w, scale_h)

            display_w = int(w * scale)
            display_h = int(h * scale)

            # 确保大小合理
            display_w = min(display_w, label_width)
            display_h = min(display_h, label_height)

            # 使用FastTransformation替代SmoothTransformation以提升性能
            scaled_pixmap = pixmap.scaled(display_w, display_h, Qt.KeepAspectRatio, Qt.FastTransformation)

            # 确保视频标签可见
            if not self.cv_video_label.isVisible():
                self.cv_video_label.show()

            # 设置图像到标签
            self.cv_video_label.setPixmap(scaled_pixmap)

        except Exception as e:
            # 只在出错时打印，避免性能损失
            print(f"显示帧时出错: {e}")
    
    def showNextFrame(self):
        """显示下一帧（优化版）"""
        if not hasattr(self, 'cv_cap') or self.cv_cap is None or not self.cv_cap.isOpened():
            if hasattr(self, 'cv_play_timer') and self.cv_play_timer.isActive():
                self.cv_play_timer.stop()
                self.play_pause_button.setText("播放")
            return

        try:
            # 检查播放器状态
            if not self.cv_cap.isOpened():
                self.cv_play_timer.stop()
                self.play_pause_button.setText("播放")
                return

            # 读取下一帧
            ret, frame = self.cv_cap.read()

            # 如果读取成功
            if ret:
                # 更新当前帧计数
                self.cv_current_frame += 1

                # 显示帧
                self.showFrame(frame)

                # 减少进度条更新频率（每10帧更新一次）
                if self.cv_current_frame % 10 == 0 and self.cv_frame_count > 0:
                    progress = int((self.cv_current_frame / self.cv_frame_count) * 100)
                    self.position_slider.setValue(progress)

                # 减少时间标签更新频率（每10帧更新一次）
                if self.cv_current_frame % 10 == 0:
                    current_time = self.cv_current_frame / self.cv_fps if self.cv_fps > 0 else 0
                    total_time = self.cv_frame_count / self.cv_fps if self.cv_fps > 0 else 0
                    self.time_label.setText(f"{self.formatTime(current_time)} / {self.formatTime(total_time)}")

            else:
                # 视频播放完毕，停止计时器并重置
                self.cv_play_timer.stop()

                # 重置视频到开始位置
                self.cv_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.cv_current_frame = 0

                # 读取第一帧但不显示，只是准备好
                ret, _ = self.cv_cap.read()

                # 更新UI
                self.play_pause_button.setText("播放")
                self.position_slider.setValue(0)
                total_time = self.cv_frame_count / self.cv_fps if self.cv_fps > 0 else 0
                self.time_label.setText(f"00:00 / {self.formatTime(total_time)}")

        except Exception as e:
            print(f"播放下一帧时出错: {e}")
            if hasattr(self, 'cv_play_timer') and self.cv_play_timer.isActive():
                self.cv_play_timer.stop()
            self.play_pause_button.setText("播放")
    
    def mediaStateChanged(self, state):
        """媒体状态改变时调用"""
        if state == QMediaPlayer.PlayingState:
            self.play_pause_button.setText("暂停")
        else:
            self.play_pause_button.setText("播放")
    
    def positionChanged(self, position):
        """播放位置改变时调用"""
        if self.media_player.duration() > 0:
            progress = int(position / self.media_player.duration() * 100)
            self.position_slider.setValue(progress)
            
            # 更新时间标签
            current_time = position / 1000  # 毫秒转秒
            total_time = self.media_player.duration() / 1000
            self.time_label.setText(f"{self.formatTime(current_time)} / {self.formatTime(total_time)}")
    
    def durationChanged(self, duration):
        """媒体时长改变时调用"""
        self.position_slider.setValue(0)
        total_time = duration / 1000  # 毫秒转秒
        self.time_label.setText(f"00:00 / {self.formatTime(total_time)}")
    
    def formatTime(self, seconds):
        """将秒数格式化为分:秒格式"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def loadFaceImages(self):
        """加载人脸图片列表到视频模式和摄像头模式"""
        try:
            face_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "input_faces")
            if not os.path.exists(face_dir):
                os.makedirs(face_dir, exist_ok=True)

            # 清空现有列表
            self.face_list.clear()
            if hasattr(self, 'video_face_list'):
                self.video_face_list.clear()

            for file in os.listdir(face_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    file_path = os.path.join(face_dir, file)

                    # 创建缩略图（只加载一次）
                    pixmap = QPixmap(file_path)
                    if pixmap.isNull():
                        continue

                    # 为摄像头模式创建项
                    item_camera = QListWidgetItem()
                    item_camera.setData(Qt.UserRole, file_path)
                    pixmap_camera = pixmap.scaled(130, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    item_camera.setIcon(QIcon(pixmap_camera))
                    item_camera.setText(os.path.basename(file_path))
                    item_camera.setSizeHint(QSize(150, 160))
                    self.face_list.addItem(item_camera)

                    # 为视频模式创建项
                    if hasattr(self, 'video_face_list'):
                        item_video = QListWidgetItem()
                        item_video.setData(Qt.UserRole, file_path)
                        pixmap_video = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        item_video.setIcon(QIcon(pixmap_video))
                        item_video.setText(os.path.basename(file_path))
                        item_video.setSizeHint(QSize(120, 130))
                        self.video_face_list.addItem(item_video)

        except Exception as e:
            print(f"加载人脸图片失败: {e}")
            QMessageBox.warning(self, "警告", f"加载人脸图片失败: {e}")
    
    def loadVideoFiles(self):
        """加载视频文件列表"""
        try:
            video_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "input_videos")
            if not os.path.exists(video_dir):
                os.makedirs(video_dir, exist_ok=True)
                
            for file in os.listdir(video_dir):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    file_path = os.path.join(video_dir, file)
                    
                    # 创建列表项
                    item = QListWidgetItem()
                    item.setData(Qt.UserRole, file_path)
                    
                    # 获取视频缩略图
                    try:
                        cap = cv2.VideoCapture(file_path)
                        ret, frame = cap.read()
                        if ret:
                            # 转换为QPixmap并设置为图标
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            h, w, c = frame.shape
                            q_img = QImage(frame.data, w, h, w * c, QImage.Format_RGB888)
                            pixmap = QPixmap.fromImage(q_img)
                            pixmap = pixmap.scaled(200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                            item.setIcon(QIcon(pixmap))
                        cap.release()
                    except Exception as e:
                        print(f"加载视频缩略图失败: {e}")
                        # 使用默认图标
                        item.setIcon(QIcon.fromTheme("video-x-generic"))
                    
                    item.setText(os.path.basename(file_path))
                    item.setSizeHint(QSize(220, 180))
                    self.video_list.addItem(item)
        except Exception as e:
            print(f"加载视频文件失败: {e}")
            QMessageBox.warning(self, "警告", f"加载视频文件失败: {e}")
    
    def selectFaceImage(self, item):
        """选择人脸图片"""
        self.selected_face_path = item.data(Qt.UserRole)
        self.statusBar().showMessage(f"已选择人脸图片: {os.path.basename(self.selected_face_path)}")
    
    def selectVideoFile(self, item):
        """选择视频文件"""
        self.selected_video_path = item.data(Qt.UserRole)
        self.statusBar().showMessage(f"已选择视频: {os.path.basename(self.selected_video_path)}")
        
        # 自动生成输出路径
        base_name = os.path.splitext(os.path.basename(self.selected_video_path))[0]
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output_videos")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}_face_swap.mp4")
        self.output_path_edit.setText(output_path)
    
    def browseOutputPath(self):
        """浏览选择输出路径"""
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output_videos")
        os.makedirs(output_dir, exist_ok=True)
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存输出视频",
            self.output_path_edit.text() or output_dir,
            "MP4文件 (*.mp4);;所有文件 (*.*)"
        )
        
        if file_path:
            if not file_path.lower().endswith('.mp4'):
                file_path += '.mp4'
            self.output_path_edit.setText(file_path)
    
    def startProcessing(self):
        """开始处理视频"""
        # 检查是否选择了必要的文件
        if not hasattr(self, 'selected_face_path'):
            QMessageBox.warning(self, "警告", "请先选择一张人脸图片")
            return
            
        if not hasattr(self, 'selected_video_path'):
            QMessageBox.warning(self, "警告", "请先选择一个视频文件")
            return
            
        output_path = self.output_path_edit.text()
        if not output_path:
            QMessageBox.warning(self, "警告", "请指定输出路径")
            return
        
        # 更新处理参数
        self.original_app.color_correction_var = self.color_correction_check.isChecked()
        self.original_app.multi_scale_var = self.multi_scale_check.isChecked()
        self.original_app.detector_var = "dlib" if self.dlib_radio.isChecked() else "opencv"
        self.original_app.swapper_var = "inswapper" if self.inswapper_radio.isChecked() else "traditional"
        self.original_app.smoothing_var = self.smooth_slider.value()
        
        # 禁用处理按钮
        self.process_button.setEnabled(False)
        self.process_button.setText("处理中...")
        self.status_label.setText("正在处理...")
        
        # 创建和启动处理线程
        self.processing_thread = VideoProcessingThread(
            self.original_app,
            self.selected_video_path,
            self.selected_face_path,
            output_path
        )
        
        # 连接信号
        self.processing_thread.progress_signal.connect(self.updateProgress)
        self.processing_thread.status_signal.connect(self.updateStatus)
        self.processing_thread.finished_signal.connect(self.processingFinished)
        self.processing_thread.error_signal.connect(self.processingError)
        
        # 启动线程
        self.processing_thread.start()
    
    def updateProgress(self, value):
        """更新进度条"""
        self.circular_progress.setValue(value)
    
    def updateStatus(self, text):
        """更新状态文本"""
        self.status_label.setText(text)
        self.statusBar().showMessage(text)
    
    def processingFinished(self, output_path):
        """处理完成后调用"""
        self.process_button.setEnabled(True)
        self.process_button.setText("开始处理")
        self.status_label.setText("处理完成!")
        
        # 加载并播放处理后的视频
        self.loadProcessedVideo(output_path)
        
        QMessageBox.information(self, "成功", "视频处理完成!")
    
    def processingError(self, error_message):
        """处理出错时调用"""
        self.process_button.setEnabled(True)
        self.process_button.setText("开始处理")
        self.status_label.setText(f"处理错误: {error_message}")
        
        QMessageBox.critical(self, "错误", f"处理视频时出错: {error_message}")
    
    def loadProcessedVideo(self, video_path):
        """加载处理后的视频到播放器"""
        try:
            if os.path.exists(video_path):
                # 停止当前播放
                self.stopPlayback()
                
                # 确保视频标签已重置
                self.cv_video_label.clear()
                self.cv_video_label.setText("<font color='#AAAAAA'>正在加载视频...</font>")
                self.cv_video_label.repaint()  # 立即刷新UI
                
                # 保存当前视频路径
                self.current_video_path = video_path
                
                # 显示加载提示
                self.statusBar().showMessage(f"正在加载视频: {os.path.basename(video_path)}")
                QApplication.processEvents()  # 确保UI更新
                
                # 使用OpenCV播放视频，不再尝试QMediaPlayer
                self.playWithOpenCV()
            else:
                QMessageBox.warning(self, "警告", f"视频文件不存在: {video_path}")
        except Exception as e:
            print(f"加载视频失败: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "警告", f"加载视频失败: {e}")
        
    def stopPlayback(self):
        """停止视频播放"""
        print("停止视频播放并清理资源...")
        # 检查是否正在使用OpenCV播放
        if hasattr(self, 'cv_play_timer'):
            # 停止OpenCV播放
            if self.cv_play_timer.isActive():
                print("停止播放计时器")
                self.cv_play_timer.stop()
            
            # 释放视频捕获
            if hasattr(self, 'cv_cap') and self.cv_cap is not None and self.cv_cap.isOpened():
                print("释放视频捕获资源")
                self.cv_cap.release()
                self.cv_cap = None
            
            # 重置UI
            print("重置播放UI状态")
            self.play_pause_button.setText("播放")
            self.position_slider.setValue(0)
            self.time_label.setText("00:00 / 00:00")
            
        # 或者使用QMediaPlayer
        elif hasattr(self, 'media_player'):
            print("停止QMediaPlayer")
            self.media_player.stop()
            self.play_pause_button.setText("播放")
            self.position_slider.setValue(0)
            self.time_label.setText("00:00 / 00:00")
        
        # 重置播放状态相关变量
        if hasattr(self, 'cv_current_frame'):
            self.cv_current_frame = 0
        
        print("视频播放已停止")

    def openVideo(self):
        """打开视频文件对话框并播放选择的视频"""
        # 默认从output_videos目录开始
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "output_videos")
        
        # 如果最近处理过视频，优先使用该目录
        if hasattr(self, 'selected_video_path') and os.path.exists(os.path.dirname(self.selected_video_path)):
            start_dir = os.path.dirname(self.selected_video_path)
        elif os.path.exists(output_dir):
            start_dir = output_dir
        else:
            # 查看input_videos目录
            input_dir = os.path.join(base_dir, "data", "input_videos")
            if os.path.exists(input_dir):
                start_dir = input_dir
            else:
                start_dir = base_dir
            
        # 打开文件对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "打开视频文件",
            start_dir,
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        )
        
        # 如果选择了文件，则加载并播放它
        if file_path and os.path.exists(file_path):
            # 先停止当前播放
            self.stopPlayback()
            
            # 重置视频标签，确保它能显示新视频
            self.cv_video_label.setText("<font color='#AAAAAA'>正在准备新视频...</font>")
            self.cv_video_label.repaint()  # 立即刷新UI
            
            # 清理旧的视频资源
            if hasattr(self, 'cv_cap') and self.cv_cap is not None:
                self.cv_cap.release()
                self.cv_cap = None
            
            # 清除定时器
            if hasattr(self, 'cv_play_timer') and self.cv_play_timer is not None:
                if self.cv_play_timer.isActive():
                    self.cv_play_timer.stop()
            
            # 保存新的视频路径
            self.current_video_path = file_path
            
            # 延迟一小段时间后加载视频，确保UI已更新
            QTimer.singleShot(100, lambda: self.loadProcessedVideo(file_path))
            
            self.statusBar().showMessage(f"已打开视频: {os.path.basename(file_path)}")
            
            # 打印调试信息
            print(f"打开新视频: {file_path}")
            print(f"视频标签大小: {self.cv_video_label.width()}x{self.cv_video_label.height()}")

    # ========== 模式切换相关方法 ==========

    def switchMode(self, mode: AppMode):
        """切换应用模式（视频/摄像头）并应用动画过渡"""
        if self.current_mode == mode:
            return  # 已经是当前模式，无需切换

        # 停止当前模式的活动
        if self.current_mode == AppMode.CAMERA_MODE:
            # 停止摄像头
            if self.camera_active:
                self.stopCamera()
        elif self.current_mode == AppMode.VIDEO_MODE:
            # 停止视频播放
            self.stopPlayback()

        # 切换模式
        old_mode = self.current_mode
        self.current_mode = mode

        # 更新按钮状态
        if mode == AppMode.VIDEO_MODE:
            self.video_mode_btn.setChecked(True)
            self.camera_mode_btn.setChecked(False)
            self.updateModeButtonStyles(AppMode.VIDEO_MODE)

            # 切换到视频控制面板
            self.control_stack.setCurrentWidget(self.video_control_panel)

            # 切换到视频控制栏
            self.control_bar_stack.setCurrentWidget(self.video_control_bar)

            self.statusBar().showMessage("已切换到视频模式")
        else:
            self.camera_mode_btn.setChecked(True)
            self.video_mode_btn.setChecked(False)
            self.updateModeButtonStyles(AppMode.CAMERA_MODE)

            # 切换到摄像头控制面板
            self.control_stack.setCurrentWidget(self.camera_control_panel)

            # 切换到摄像头控制栏
            self.control_bar_stack.setCurrentWidget(self.camera_control_bar)

            self.statusBar().showMessage("已切换到摄像头模式")

        # 应用动画过渡效果
        self.animateModeTransition(old_mode, mode)

    def updateModeButtonStyles(self, active_mode: AppMode):
        """更新模式切换按钮的样式"""
        active_style = """
            QPushButton {
                background-color: #4287f5;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:checked {
                background-color: #5a9cff;
                border: 2px solid #85b7ff;
            }
            QPushButton:hover {
                background-color: #5a9cff;
            }
        """

        inactive_style = """
            QPushButton {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #444444;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:checked {
                background-color: #4287f5;
                border: 2px solid #85b7ff;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
        """

        if active_mode == AppMode.VIDEO_MODE:
            self.video_mode_btn.setStyleSheet(active_style)
            self.camera_mode_btn.setStyleSheet(inactive_style)
        else:
            self.video_mode_btn.setStyleSheet(inactive_style)
            self.camera_mode_btn.setStyleSheet(active_style)

    def animateModeTransition(self, old_mode: AppMode, new_mode: AppMode):
        """应用模式切换的动画过渡效果"""
        try:
            # 创建淡入淡出动画
            self.effect_widget = QWidget(self)
            self.effect_widget.setGeometry(self.rect())
            self.effect_widget.setStyleSheet("background-color: black;")

            # 创建透明度效果
            self.opacity_effect = QGraphicsOpacityEffect(self.effect_widget)
            self.effect_widget.setGraphicsEffect(self.opacity_effect)
            self.opacity_effect.setOpacity(0.0)
            self.effect_widget.show()

            # 淡入动画
            fade_in = QPropertyAnimation(self.opacity_effect, b"opacity")
            fade_in.setDuration(150)
            fade_in.setStartValue(0.0)
            fade_in.setEndValue(0.7)

            # 淡出动画
            fade_out = QPropertyAnimation(self.opacity_effect, b"opacity")
            fade_out.setDuration(150)
            fade_out.setStartValue(0.7)
            fade_out.setEndValue(0.0)

            # 顺序执行
            self.sequential_animation = QSequentialAnimationGroup()
            self.sequential_animation.addAnimation(fade_in)
            self.sequential_animation.addAnimation(fade_out)

            # 动画结束后隐藏效果部件
            self.sequential_animation.finished.connect(self.cleanupAnimationEffect)

            self.sequential_animation.start()

        except Exception as e:
            print(f"动画过渡失败: {e}")
            # 即使动画失败，模式切换也应该继续

    def cleanupAnimationEffect(self):
        """清理动画效果"""
        try:
            if hasattr(self, 'effect_widget') and self.effect_widget is not None:
                self.effect_widget.hide()
                self.effect_widget.deleteLater()
        except Exception as e:
            print(f"清理动画效果时出错: {e}")

    # ========== 摄像头控制相关方法 ==========

    def toggleCamera(self):
        """开启/关闭摄像头"""
        if self.camera_active:
            self.stopCamera()
        else:
            self.startCamera()

    def startCamera(self):
        """启动摄像头"""
        try:
            # 检查是否选择了人脸图片
            if not hasattr(self, 'selected_face_path'):
                QMessageBox.warning(self, "警告", "请先选择一张人脸图片")
                return

            # 停止视频播放（如果正在播放）
            self.stopPlayback()
            self.cv_video_label.clear()

            # 创建摄像头线程
            self.camera_thread = CameraProcessingThread(
                self.original_app,
                self.selected_face_path
            )

            # 连接信号
            self.camera_thread.frame_ready.connect(self.displayCameraFrame)
            self.camera_thread.status_signal.connect(self.updateCameraStatus)
            self.camera_thread.error_signal.connect(self.handleCameraError)

            # 启动摄像头
            self.camera_thread.start_camera()
            self.camera_active = True

            # 更新UI
            self.camera_toggle_btn.setText("关闭摄像头")
            self.camera_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #c82333;
                    box-shadow: 0 4px 8px rgba(220, 53, 69, 0.3);
                }
            """)

            # 更新状态指示器
            self.camera_status_indicator.setStyleSheet("""
                QLabel {
                    background-color: #28a745;
                    border-radius: 6px;
                    border: 2px solid #34ce57;
                    box-shadow: 0 0 8px #28a745;
                }
            """)

            # 更新状态文本
            self.camera_bar_status_label.setText("摄像头运行中")
            self.camera_bar_status_label.setStyleSheet("""
                QLabel {
                    color: #28a745;
                    font-size: 13px;
                    font-weight: bold;
                    padding: 4px 10px;
                    background-color: #1a2e1a;
                    border: 1px solid #28a745;
                    border-radius: 4px;
                }
            """)

            self.snapshot_btn.setEnabled(True)
            self.statusBar().showMessage("摄像头已启动")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动摄像头失败: {e}")
            import traceback
            traceback.print_exc()

    def stopCamera(self):
        """停止摄像头"""
        try:
            if self.camera_thread and self.camera_active:
                self.camera_thread.stop_camera()
                self.camera_active = False

                # 更新UI
                self.camera_toggle_btn.setText("开启摄像头")
                self.camera_toggle_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #28a745;
                        color: white;
                        border: none;
                        padding: 8px 16px;
                        border-radius: 4px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #218838;
                    }
                """)

                # 清空显示区域
                self.cv_video_label.setText("<font color='#888888'>摄像头已关闭</font>")

                # 重置状态指示器
                self.camera_status_indicator.setStyleSheet("""
                    QLabel {
                        background-color: #666666;
                        border-radius: 6px;
                        border: 2px solid #888888;
                    }
                """)

                # 重置状态文本
                self.camera_bar_status_label.setText("摄像头未启动")
                self.camera_bar_status_label.setStyleSheet("""
                    QLabel {
                        color: #888888;
                        font-size: 13px;
                        padding: 4px 10px;
                        background-color: #1a1a1a;
                        border-radius: 4px;
                    }
                """)

                # 清空分辨率显示
                if hasattr(self, 'camera_resolution_label'):
                    self.camera_resolution_label.setText("")

                self.statusBar().showMessage("摄像头已关闭")

        except Exception as e:
            print(f"停止摄像头时出错: {e}")

    def displayCameraFrame(self, frame):
        """显示摄像头捕获的帧（全屏显示，不缩放）"""
        try:
            if frame is None:
                return

            # 转换颜色空间
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 获取视频原始尺寸
            h, w, c = frame_rgb.shape

            # 转换为QImage
            q_img = QImage(frame_rgb.data, w, h, w * c, QImage.Format_RGB888)

            # 创建QPixmap
            pixmap = QPixmap.fromImage(q_img)

            # 直接显示原始大小，不进行缩放
            self.cv_video_label.setPixmap(pixmap)

            # 更新分辨率显示
            if hasattr(self, 'camera_resolution_label'):
                self.camera_resolution_label.setText(f"{w}x{h}")

        except Exception as e:
            # 减少日志输出
            pass

    def updateCameraStatus(self, status: str):
        """更新摄像头状态信息"""
        self.camera_status_label.setText(status)
        self.statusBar().showMessage(status)

    def handleCameraError(self, error: str):
        """处理摄像头错误"""
        QMessageBox.critical(self, "摄像头错误", error)
        self.stopCamera()

    def takeSnapshot(self):
        """拍照并保存当前帧"""
        try:
            if not self.camera_active or not hasattr(self, 'camera_thread'):
                QMessageBox.warning(self, "警告", "摄像头未启动")
                return

            # 获取当前显示的帧
            pixmap = self.cv_video_label.pixmap()
            if pixmap is None or pixmap.isNull():
                QMessageBox.warning(self, "警告", "无法捕获当前帧")
                return

            # 选择保存位置
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "input_faces")
            os.makedirs(output_dir, exist_ok=True)

            timestamp = int(time.time())
            default_filename = f"snapshot_{timestamp}.png"
            default_path = os.path.join(output_dir, default_filename)

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存快照",
                default_path,
                "PNG文件 (*.png);;JPEG文件 (*.jpg);;所有文件 (*.*)"
            )

            if file_path:
                # 保存图像
                pixmap.save(file_path)
                self.statusBar().showMessage(f"快照已保存: {os.path.basename(file_path)}")

                # 询问是否重新加载人脸列表
                reply = QMessageBox.question(
                    self,
                    "重新加载",
                    "快照已保存。是否重新加载人脸列表？",
                    QMessageBox.Yes | QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    self.face_list.clear()
                    self.loadFaceImages()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"拍照失败: {e}")
            import traceback
            traceback.print_exc()

    def closeEvent(self, event):
        """应用程序关闭事件，清理资源"""
        try:
            # 停止视频播放
            self.stopPlayback()

            # 停止摄像头
            if self.camera_active:
                self.stopCamera()

            # 如果有处理线程正在运行，停止它
            if hasattr(self, 'processing_thread') and self.processing_thread.isRunning():
                self.processing_thread.wait(1000)  # 等待最多1秒
            
            # 释放OpenCV视频捕获
            if hasattr(self, 'cv_cap') and self.cv_cap is not None and self.cv_cap.isOpened():
                self.cv_cap.release()
            
            # 停止所有定时器
            if hasattr(self, 'cv_play_timer') and self.cv_play_timer.isActive():
                self.cv_play_timer.stop()
                
        except Exception as e:
            print(f"清理资源时出错: {e}")
            import traceback
            traceback.print_exc()
            
        # 继续默认的关闭事件
        super().closeEvent(event)

    def dragEnterEvent(self, event):
        """处理拖入事件"""
        if event.mimeData().hasUrls():
            # 检查是否是视频文件
            urls = event.mimeData().urls()
            for url in urls:
                file_path = url.toLocalFile()
                if os.path.isfile(file_path) and file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    event.acceptProposedAction()
                    return
    
    def dropEvent(self, event):
        """处理放置事件"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                file_path = url.toLocalFile()
                if os.path.isfile(file_path) and file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    self.loadProcessedVideo(file_path)
                    self.statusBar().showMessage(f"已加载拖放的视频: {os.path.basename(file_path)}")
                    break
            event.acceptProposedAction()

    def changePlaybackSpeed(self, speed_text):
        """更改视频播放速度"""
        if not hasattr(self, 'speed_options'): # 确保speed_options已初始化
            return
            
        self.playback_speed_factor = self.speed_options.get(speed_text, 1.0)
        
        if hasattr(self, 'media_player') and self.media_player.isSeekable() and self.media_player.state() == QMediaPlayer.PlayingState:
            # QMediaPlayer支持直接设置播放速率
            self.media_player.setPlaybackRate(self.playback_speed_factor)
        elif hasattr(self, 'cv_play_timer') and self.cv_play_timer.isActive():
            # 对于OpenCV，我们需要停止并用新的延迟重新启动计时器
            self.cv_play_timer.stop()
            if self.cv_fps > 0:
                delay = int(1000 / (self.cv_fps * self.playback_speed_factor))
                if delay <= 0: # 防止延迟为0或负数
                    delay = 1 # 设置一个最小延迟
                self.cv_play_timer.start(delay)
            else:
                # 如果fps为0，则无法调整速度，使用基于播放速度因子的默认延迟
                default_fps = 30
                delay = int(1000 / (default_fps * self.playback_speed_factor))
                if delay <= 0:
                    delay = 1
                self.cv_play_timer.start(delay)
        
        print(f"播放速度已更改为: {self.playback_speed_factor}x")

def main():
    app = QApplication(sys.argv)
    
    # 尝试使用QDarkStyle主题
    try:
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        print("已应用暗色主题")
    except ImportError as e:
        print(f"未能加载暗色主题: {e}")
        # 使用内置的Fusion风格
        app.setStyle("Fusion")
        # 创建暗色调色板
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        app.setPalette(dark_palette)
    
    # 启动应用程序
    window = EnhancedFaceSwapUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 