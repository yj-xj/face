import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox, QProgressBar, QSplitter, QScrollArea,
    QListWidget, QListWidgetItem, QFrame, QGridLayout, QGroupBox, QLineEdit,
    QComboBox, QCheckBox, QRadioButton, QStackedWidget, QButtonGroup, QSlider,
    QInputDialog
)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPalette, QColor, QFont
from PyQt5.QtCore import Qt, QSize, QTimer, QUrl, pyqtSignal, QThread
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

# 检查是否需要PyQt5 Multimedia插件
try:
    from PyQt5.QtMultimediaWidgets import QVideoWidget
except ImportError:
    QMessageBox.critical(None, "错误", "未找到PyQt5 Multimedia插件，请确保安装了PyQt5.QtMultimedia")
    sys.exit(1)

# 导入原始的人脸替换功能
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from face_swap import FaceSwapApp as OriginalFaceSwapApp

class VideoProcessingThread(QThread):
    """视频处理线程，防止UI卡顿"""
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
            
            # 创建新的更新方法，将进度和状态信号传递给UI
            def new_update_progress(value, text=None):
                self.progress_signal.emit(int(value))
                # 调用原始方法记录日志等操作
                original_update_progress(value, text)
                
            def new_update_status(text):
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
        
        # 初始化原始FaceSwapApp的实例
        self.original_app = OriginalFaceSwapApp(None)
        # 由于原始应用需要root参数，在这里手动设置一些关键属性
        self.original_app.root = None  # 明确设置为None
        # 初始化必要的变量
        self.original_app.swap_method_var = "advanced"  # 默认使用高级替换方法
        self.original_app.color_correction_var = True   # 默认启用颜色校正
        self.original_app.multi_scale_var = True       # 默认使用多尺度检测
        self.original_app.detector_var = "dlib"        # 默认使用dlib检测器
        self.original_app.swapper_var = "inswapper"    # 默认使用inswapper
        self.original_app.smoothing_var = 50           # 默认平滑度
        
        # 设置窗口属性
        self.setWindowTitle("人脸替换应用")
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
        
        # 创建顶部标题
        title_label = QLabel("人脸替换应用")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #4287f5;
            margin-bottom: 10px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
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
        
        # 视频控制栏
        control_layout = QHBoxLayout()
        
        # 添加打开视频按钮
        open_video_btn = QPushButton("打开视频")
        open_video_btn.setIcon(QIcon.fromTheme("document-open"))
        open_video_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #444444;
                border-radius: 3px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
                border: 1px solid #555555;
            }
            QPushButton:pressed {
                background-color: #222222;
            }
        """)
        open_video_btn.clicked.connect(self.openVideo)
        control_layout.addWidget(open_video_btn)
        
        self.play_button = QPushButton("播放")
        self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #444444;
                border-radius: 3px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
                border: 1px solid #555555;
            }
            QPushButton:pressed {
                background-color: #222222;
            }
        """)
        self.play_button.clicked.connect(self.togglePlayback)
        control_layout.addWidget(self.play_button)
        
        # 添加停止按钮
        self.stop_button = QPushButton("停止")
        self.stop_button.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #444444;
                border-radius: 3px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
                border: 1px solid #555555;
            }
            QPushButton:pressed {
                background-color: #222222;
            }
        """)
        self.stop_button.clicked.connect(self.stopPlayback)
        control_layout.addWidget(self.stop_button)
        
        self.position_slider = QProgressBar()
        self.position_slider.setTextVisible(False)
        self.position_slider.setRange(0, 100)
        self.position_slider.setValue(0)
        self.position_slider.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444444;
                border-radius: 3px;
                background-color: #2a2a2a;
                height: 15px;
            }
            QProgressBar::chunk {
                background-color: #4287f5;
                width: 1px;
            }
        """)
        
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("color: white;")
        
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.position_slider, 1)
        control_layout.addWidget(self.time_label)
        
        left_layout.addLayout(control_layout)
        
        # 创建右侧控制部分
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 创建文件选择部分
        input_group = QGroupBox("输入文件")
        input_layout = QVBoxLayout(input_group)
        
        # 人脸图片列表
        face_label = QLabel("选择人脸图片:")
        face_label.setStyleSheet("font-weight: bold;")
        input_layout.addWidget(face_label)
        
        self.face_list = QListWidget()
        self.face_list.setFixedHeight(220)
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
        input_layout.addWidget(self.face_list)
        
        # 视频文件列表
        video_label = QLabel("选择视频:")
        video_label.setStyleSheet("font-weight: bold;")
        input_layout.addWidget(video_label)
        
        self.video_list = QListWidget()
        self.video_list.setFixedHeight(240)
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
        input_layout.addWidget(self.video_list)
        
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
        
        input_layout.addLayout(output_layout)
        
        right_layout.addWidget(input_group)
        
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
        
        right_layout.addWidget(advanced_group)
        
        # 处理按钮和状态
        process_group = QGroupBox("处理控制")
        process_layout = QVBoxLayout(process_group)
        
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
        
        process_layout.addLayout(progress_layout)
        
        right_layout.addWidget(process_group)
        right_layout.addStretch()
        
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
        # 检查是否正在使用OpenCV播放
        if hasattr(self, 'cv_play_timer') and hasattr(self, 'cv_cap') and self.cv_cap is not None and self.cv_cap.isOpened():
            if self.cv_play_timer.isActive():
                # 暂停OpenCV播放
                self.cv_play_timer.stop()
                self.play_button.setText("播放")
            else:
                # 继续OpenCV播放
                if self.cv_fps <= 0:
                    self.cv_fps = 30  # 默认30fps
                delay = int(1000 / self.cv_fps)
                self.cv_play_timer.start(delay)
                self.play_button.setText("暂停")
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
            if self.cv_cap is None or not self.cv_cap.isOpened():
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
            delay = int(1000 / self.cv_fps)
            
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
            print(f"开始定时播放，间隔: {delay}ms")
            self.cv_play_timer.start(delay)
            
            # 更新UI状态
            self.play_button.setText("暂停")
            self.statusBar().showMessage(f"正在使用OpenCV播放: {os.path.basename(self.current_video_path)}")
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"OpenCV播放失败: {e}")
            import traceback
            traceback.print_exc()
    
    def showFrame(self, frame):
        """显示一帧视频"""
        try:
            if frame is None:
                print("警告: 收到空帧")
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
                    print(f"使用父容器尺寸: {label_width}x{label_height}")
                else:
                    label_width = 800
                    label_height = 600
                    print("使用默认尺寸: 800x600")
            
            # 输出标签尺寸便于调试
            print(f"视频标签大小: {label_width}x{label_height}")
            
            # 转换为QImage
            h, w, c = frame_rgb.shape
            q_img = QImage(frame_rgb.data, w, h, w * c, QImage.Format_RGB888)
            
            # 创建QPixmap并调整大小以适应视频标签
            pixmap = QPixmap.fromImage(q_img)
            
            # 计算缩放比例，保持视频的原始宽高比
            # 由于这个视频是竖屏视频，我们调整策略，以高度为主来缩放
            scale_w = label_width / w
            scale_h = label_height / h
            
            # 如果视频是横向的（宽大于高）
            if w > h:
                scale = min(scale_w, scale_h)
            else:
                # 如果是竖向视频（高大于宽），优先填充高度
                scale = scale_h
            
            # 计算实际显示大小
            display_w = int(w * scale)
            display_h = int(h * scale)
            
            # 确保大小合理
            display_w = min(display_w, label_width)
            display_h = min(display_h, label_height)
            
            # 缩放图像
            scaled_pixmap = pixmap.scaled(display_w, display_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # 确保视频标签可见
            if not self.cv_video_label.isVisible():
                print("视频标签不可见，设为可见")
                self.cv_video_label.show()
            
            # 设置图像到标签
            self.cv_video_label.setPixmap(scaled_pixmap)
            
            # 每20帧输出一次日志，避免控制台刷屏
            if hasattr(self, 'cv_current_frame') and self.cv_current_frame % 20 == 0:
                print(f"显示帧 {self.cv_current_frame}/{self.cv_frame_count}: 图像大小 {scaled_pixmap.width()}x{scaled_pixmap.height()}")
            
        except Exception as e:
            print(f"显示帧时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def showNextFrame(self):
        """显示下一帧"""
        if not hasattr(self, 'cv_cap') or self.cv_cap is None or not self.cv_cap.isOpened():
            if hasattr(self, 'cv_play_timer') and self.cv_play_timer.isActive():
                print("视频源不可用，停止播放")
                self.cv_play_timer.stop()
                self.play_button.setText("播放")
            return
        
        try:
            # 检查播放器状态
            if self.cv_cap is None or not self.cv_cap.isOpened():
                print("视频捕获已关闭，停止播放")
                self.cv_play_timer.stop()
                self.play_button.setText("播放")
                return
            
            # 读取下一帧
            ret, frame = self.cv_cap.read()
            
            # 如果读取成功
            if ret:
                # 更新当前帧计数
                self.cv_current_frame += 1
                
                # 显示帧
                self.showFrame(frame)
                
                # 更新进度条
                if self.cv_frame_count > 0:
                    progress = int((self.cv_current_frame / self.cv_frame_count) * 100)
                    self.position_slider.setValue(progress)
                
                # 更新时间标签
                current_time = self.cv_current_frame / self.cv_fps if self.cv_fps > 0 else 0
                total_time = self.cv_frame_count / self.cv_fps if self.cv_fps > 0 else 0
                self.time_label.setText(f"{self.formatTime(current_time)} / {self.formatTime(total_time)}")
                
            else:
                # 视频播放完毕，停止计时器并重置
                print("视频播放完毕")
                self.cv_play_timer.stop()
                
                # 重置视频到开始位置
                self.cv_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.cv_current_frame = 0
                
                # 读取第一帧但不显示，只是准备好
                ret, _ = self.cv_cap.read()
                
                # 更新UI
                self.play_button.setText("播放")
                self.position_slider.setValue(0)
                total_time = self.cv_frame_count / self.cv_fps if self.cv_fps > 0 else 0
                self.time_label.setText(f"00:00 / {self.formatTime(total_time)}")
                
        except Exception as e:
            print(f"播放下一帧时出错: {e}")
            if hasattr(self, 'cv_play_timer') and self.cv_play_timer.isActive():
                self.cv_play_timer.stop()
            self.play_button.setText("播放")
            import traceback
            traceback.print_exc()
    
    def mediaStateChanged(self, state):
        """媒体状态改变时调用"""
        if state == QMediaPlayer.PlayingState:
            self.play_button.setText("暂停")
        else:
            self.play_button.setText("播放")
    
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
        """加载人脸图片列表"""
        try:
            face_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "input_faces")
            if not os.path.exists(face_dir):
                os.makedirs(face_dir, exist_ok=True)
                
            for file in os.listdir(face_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    file_path = os.path.join(face_dir, file)
                    
                    # 创建列表项
                    item = QListWidgetItem()
                    item.setData(Qt.UserRole, file_path)
                    
                    # 创建缩略图
                    pixmap = QPixmap(file_path)
                    if not pixmap.isNull():
                        pixmap = pixmap.scaled(130, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        item.setIcon(QIcon(pixmap))
                        item.setText(os.path.basename(file_path))
                        item.setSizeHint(QSize(150, 160))
                        self.face_list.addItem(item)
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
            self.play_button.setText("播放")
            self.position_slider.setValue(0)
            self.time_label.setText("00:00 / 00:00")
            
        # 或者使用QMediaPlayer
        elif hasattr(self, 'media_player'):
            print("停止QMediaPlayer")
            self.media_player.stop()
            self.play_button.setText("播放")
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

    def closeEvent(self, event):
        """应用程序关闭事件，清理资源"""
        try:
            # 停止视频播放
            self.stopPlayback()
            
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