import os
import sys
import time  # 添加time模块导入

# 设置环境变量，解决OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置Python环境路径
python_path = r'E:\anaconda\envs\face_swap'
if python_path not in sys.path:
    sys.path.append(python_path)

# 设置Python库路径
lib_path = os.path.join(python_path, 'Lib')
site_packages = os.path.join(lib_path, 'site-packages')
if site_packages not in sys.path:
    sys.path.append(site_packages)

# 将当前目录添加到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 解决Tcl/Tk版本冲突问题
os.environ['TCL_LIBRARY'] = os.path.join(python_path, 'Library', 'lib', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(python_path, 'Library', 'lib', 'tk8.6')
os.environ['TCL_IGNORE_VERSION_CHECK'] = '1'

# 自动修复init.tcl文件，解决tkinter初始化问题
tcl_lib_path = os.path.join(python_path, 'Library', 'lib', 'tcl8.6') 
init_tcl_path = os.path.join(tcl_lib_path, 'init.tcl')
orig_tk_path = os.path.join(python_path, 'Library', 'lib', 'tk8.6')
tk_tcl_path = os.path.join(orig_tk_path, 'tk.tcl')

# 检查并修复init.tcl文件
if not os.path.exists(tcl_lib_path):
    os.makedirs(tcl_lib_path, exist_ok=True)
    print(f"已创建Tcl库目录: {tcl_lib_path}")

if not os.path.exists(orig_tk_path):
    os.makedirs(orig_tk_path, exist_ok=True)
    print(f"已创建Tk库目录: {orig_tk_path}")

# 修复init.tcl
if not os.path.exists(init_tcl_path):
    try:
        # 尝试从anaconda基础环境中复制
        base_tcl_path = os.path.join(os.path.dirname(os.path.dirname(python_path)), 'Library', 'lib', 'tcl8.6', 'init.tcl')
        if os.path.exists(base_tcl_path):
            import shutil
            shutil.copy(base_tcl_path, init_tcl_path)
            print(f"已修复init.tcl文件: {init_tcl_path}")
            
            # 修改版本号
            with open(init_tcl_path, 'r') as f:
                content = f.read()
            # 将8.6.14替换为8.6.12
            content = content.replace('package require -exact Tcl 8.6.14', 'package require -exact Tcl 8.6.12')
            with open(init_tcl_path, 'w') as f:
                f.write(content)
            print("已修改 init.tcl 文件以适配 Tcl 8.6.12 版本")
        else:
            # 创建最小init.tcl
            with open(init_tcl_path, 'w') as f:
                f.write("""
# Minimal init.tcl for Tcl 8.6
if {[info commands package] == ""} {
    proc package {args} {
        return
    }
}
package require -exact Tcl 8.6.12
""")
            print(f"已创建最小化init.tcl文件: {init_tcl_path}")
    except Exception as e:
        print(f"修复init.tcl时出错: {e}")

# 修复tk.tcl
if not os.path.exists(tk_tcl_path):
    try:
        # 尝试从anaconda基础环境中复制
        base_tk_path = os.path.join(os.path.dirname(os.path.dirname(python_path)), 'Library', 'lib', 'tk8.6', 'tk.tcl')
        if os.path.exists(base_tk_path):
            import shutil
            shutil.copy(base_tk_path, tk_tcl_path)
            
            # 修改版本号
            with open(tk_tcl_path, 'r') as f:
                content = f.read()
            # 将8.6.14替换为8.6.12
            content = content.replace('package require -exact Tcl 8.6.14', 'package require -exact Tcl 8.6.12')
            content = content.replace('package require -exact Tk 8.6.14', 'package require -exact Tk 8.6.12')
            with open(tk_tcl_path, 'w') as f:
                f.write(content)
            print("已修改 tk.tcl 文件以适配 Tk 8.6.12 版本")
        else:
            # 创建最小tk.tcl
            with open(tk_tcl_path, 'w') as f:
                f.write("""
# Minimal tk.tcl for Tk 8.6
if {[info commands namespace] == ""} {
    proc namespace {args} {
        return
    }
}
namespace eval ::tk {
    namespace eval scaling {
        variable factor 1.0
    }
}
""")
            print(f"已创建最小化tk.tcl文件: {tk_tcl_path}")
    except Exception as e:
        print(f"修复tk.tcl时出错: {e}")

# 导入SSL验证忽略模块
try:
    from ignore_ssl_warnings import patch_insightface_download
    # 修补insightface下载功能
    patch_insightface_download()
except ImportError:
    print("警告：无法导入ignore_ssl_warnings模块")

try:
    # 首先导入基础模块
    import cv2
    import numpy as np
    import dlib
    
    # 导入GUI相关模块
    from PIL import Image, ImageTk
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    import threading
    
    # 导入视频处理相关模块
    import imageio_ffmpeg
    import moviepy.config as mpconfig
    import moviepy
    
    # 设置moviepy的ffmpeg路径
    moviepy.config.FFMPEG_BINARY = imageio_ffmpeg.get_ffmpeg_exe()
    moviepy.config.IMAGEMAGICK_BINARY = "convert"
    if hasattr(moviepy.config, 'PYTHON_BINARY'):
        moviepy.config.PYTHON_BINARY = os.path.join(python_path, 'python.exe')
    
    import concurrent.futures
    from moviepy.editor import VideoFileClip
    import logging
    
    # 尝试导入insightface库
    try:
        import insightface
        import onnxruntime
        INSIGHTFACE_AVAILABLE = True
        print("成功导入InsightFace模块")
    except ImportError as e:
        INSIGHTFACE_AVAILABLE = False
        print(f"导入InsightFace模块时出错: {e}")
        print("请安装InsightFace: pip install insightface onnx onnxruntime")
        
    print("所有基础模块已成功导入")
    
except ImportError as e:
    print(f"导入模块时出错: {e}")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         "logs", f"face_swap_{time.strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("face_swap")

class FaceSwapApp:
    def __init__(self, root):
        # 防止局部变量遮蔽全局变量
        global cv2, dlib, INSIGHTFACE_AVAILABLE
        
        self.root = root
        
        # 设置一些基本颜色，即使没有UI也需要
        self.primary_color = "#4287f5"     # 主色调蓝色
        self.secondary_color = "#f0f0f0"   # 背景色浅灰
        self.success_color = "#4CAF50"     # 成功色绿色
        self.warning_color = "#FF9800"     # 警告色橙色
        self.accent_color = "#2962FF"      # 强调色深蓝
        self.text_color = "#212121"        # 文字色深灰
        
        # 如果root不为None，则设置标题和大小
        if self.root is not None:
        self.root.title("人脸替换应用 - InsightFace版")
        self.root.geometry("1200x800")
        
            # 设置应用程序图标和全局字体
            self.set_app_appearance()
        
        # 存储路径
        self.video_path = ""
        self.face_images = []
        self.selected_face_index = -1
        self.output_path = ""
        
        # 设置默认文件夹路径
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_folder = os.path.join(self.base_dir, "data")
        self.output_folder = os.path.join(self.base_dir, "output_videos")
        self.models_folder = os.path.join(self.base_dir, "models")
        self.logs_folder = os.path.join(self.base_dir, "logs")
        
        # 初始化视频播放相关变量
        self.video_player = None
        self.is_playing = False
        self.current_frame = 0
        self.video_frames = []
        self.play_thread = None
        self.duration = 0  # 初始化视频时长
        self.fps = 0  # 初始化帧率
        self.total_frames = 0  # 初始化总帧数
        
        # 确保文件夹存在
        for folder in [self.data_folder, self.output_folder, self.models_folder, self.logs_folder]:
            os.makedirs(folder, exist_ok=True)
        
        # 初始化模型路径
        self.cascade_path = os.path.join(self.models_folder, "haarcascade_frontalface_default.xml")
        self.predictor_path = os.path.join(self.models_folder, "shape_predictor_68_face_landmarks.dat")
        self.inswapper_path = os.path.join(self.models_folder, "inswapper_128.onnx")
        
        # 检查并初始化OpenCV级联分类器
        self.init_cascade_classifier()
        
        # 初始化dlib人脸检测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None
        
        # 检查并加载dlib特征点预测器
        if os.path.exists(self.predictor_path):
            try:
                self.predictor = dlib.shape_predictor(self.predictor_path)
                logger.info("成功加载特征点预测模型")
            except Exception as e:
                logger.error(f"加载特征点预测模型失败: {e}")
        else:
            logger.warning(f"特征点预测模型文件不存在: {self.predictor_path}")
        
        # 初始化InsightFace模型
        self.face_analyser = None
        self.inswapper = None
        
        if INSIGHTFACE_AVAILABLE:
            try:
                if os.path.exists(self.inswapper_path):
                    # 初始化face_analyser用于检测和提取面部特征
                    # 添加verify=False参数禁用SSL验证，并设置允许本地模型
                    os.environ['INSIGHTFACE_ALLOW_LOCAL_MODEL'] = '1'
                    # 使用try-except确保初始化不会因为SSL问题而失败
                    try:
                        self.face_analyser = insightface.app.FaceAnalysis(
                            name="buffalo_l", 
                            root=self.models_folder,
                            providers=['CPUExecutionProvider'],
                            allowed_modules=['detection', 'recognition']
                        )
                        self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
                    except Exception as e:
                        logger.warning(f"使用正常方式初始化face_analyser失败: {e}")
                        # 尝试直接加载buffalo_l目录下的模型
                        buffalo_dir = os.path.join(self.models_folder, 'buffalo_l')
                        if os.path.exists(buffalo_dir):
                            try:
                                self.face_analyser = insightface.app.FaceAnalysis(
                                    name="buffalo_l", 
                                    root=self.models_folder,
                                    providers=['CPUExecutionProvider'],
                                    allowed_modules=['detection', 'recognition'],
                                    download=False  # 禁止下载
                                )
                                self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
                                logger.info("成功从本地加载buffalo_l模型")
                            except Exception as e2:
                                logger.error(f"从本地加载buffalo_l模型失败: {e2}")
                                import traceback
                                traceback.print_exc()
                        else:
                            logger.error(f"buffalo_l目录不存在: {buffalo_dir}")
                    
                    # 初始化inswapper模型
                    if os.path.exists(self.inswapper_path):
                        try:
                            self.inswapper = insightface.model_zoo.get_model(
                                self.inswapper_path, 
                                providers=['CPUExecutionProvider'],
                                download=False,
                                download_zip=False
                            )
                            logger.info(f"成功加载InsightFace模型: {self.inswapper_path}")
                        except Exception as e:
                            logger.error(f"加载InsightFace模型失败: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        logger.warning(f"InsightFace模型文件不存在: {self.inswapper_path}")
                        if self.root is not None:
                        self.show_model_download_guide("inswapper")
                else:
                    logger.warning(f"InsightFace模型文件不存在: {self.inswapper_path}")
                    if self.root is not None:
                    self.show_model_download_guide("inswapper")
            except Exception as e:
                logger.error(f"初始化InsightFace模型时出错: {e}")
                if self.root is not None:
                self.show_model_download_guide("inswapper")
                import traceback
                traceback.print_exc()
        else:
            logger.warning("InsightFace模块不可用，请安装: pip install insightface onnx onnxruntime")
        
        # 初始化高级选项
        self.detection_confidence = 0.5  # 人脸检测置信度阈值
        self.multi_scale_detection = True  # 启用多尺度检测
        
        # 创建UI - 只有当root不为None时才创建
        if self.root is not None:
        self.create_ui()
        
        # 自动加载数据文件夹中的视频和图片
        self.load_data_folder()
        
        # 检查模型文件并提示用户
        self.check_required_models()
    
    def set_app_appearance(self):
        """设置应用程序的全局外观"""
        # 配置ttk样式
        self.style = ttk.Style()
        
        # 尝试使用不同的主题
        available_themes = self.style.theme_names()
        if 'clam' in available_themes:
            self.style.theme_use('clam')
        elif 'vista' in available_themes:
            self.style.theme_use('vista')
        
        # 定义应用程序的配色方案
        self.primary_color = "#4287f5"     # 主色调蓝色
        self.secondary_color = "#f0f0f0"   # 背景色浅灰
        self.success_color = "#4CAF50"     # 成功色绿色
        self.warning_color = "#FF9800"     # 警告色橙色
        self.accent_color = "#2962FF"      # 强调色深蓝
        self.text_color = "#212121"        # 文字色深灰
        
        # 配置全局字体
        default_font = ('微软雅黑', 10)
        title_font = ('微软雅黑', 12, 'bold')
        button_font = ('微软雅黑', 10, 'bold')
        
        # 配置各种组件样式
        self.style.configure('TButton', font=button_font, background=self.primary_color)
        self.style.configure('Success.TButton', font=button_font, background=self.success_color)
        self.style.configure('TLabel', font=default_font, background=self.secondary_color)
        self.style.configure('Title.TLabel', font=title_font, background=self.secondary_color)
        self.style.configure('TFrame', background=self.secondary_color)
        self.style.configure('TEntry', font=default_font)
        self.style.configure('TCheckbutton', font=default_font, background=self.secondary_color)
        self.style.configure('TRadiobutton', font=default_font, background=self.secondary_color)
        
        # 设置滚动条样式
        self.style.configure('TScrollbar', background=self.primary_color, troughcolor=self.secondary_color, 
                            borderwidth=0, arrowsize=14)
        
        # 设置进度条样式
        self.style.configure('Horizontal.TProgressbar', background=self.primary_color, 
                            troughcolor=self.secondary_color, borderwidth=0)
        
        # 设置根窗口背景色
        self.root.configure(bg=self.secondary_color)
    
    def init_cascade_classifier(self):
        """初始化OpenCV级联分类器"""
        self.face_cascade = None
        
        # 如果文件不存在，尝试从OpenCV安装目录查找
        if not os.path.exists(self.cascade_path):
            try:
                # 尝试从OpenCV安装目录查找
                opencv_haarcascade = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
                if os.path.exists(opencv_haarcascade):
                    import shutil
                    shutil.copy(opencv_haarcascade, self.cascade_path)
                    logger.info(f"已从OpenCV复制级联分类器文件到: {self.cascade_path}")
            except (ImportError, AttributeError) as e:
                logger.error(f"查找OpenCV级联分类器出错: {e}")
                logger.warning(f"未找到级联分类器文件: {self.cascade_path}")
                logger.info("请从https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml下载")
        
        # 使用文件路径初始化分类器
        try:
            if os.path.exists(self.cascade_path):
                self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
                if self.face_cascade.empty():
                    logger.warning(f"无法加载级联分类器，文件可能损坏: {self.cascade_path}")
                    self.face_cascade = None
                else:
                    logger.info("成功加载级联分类器")
            else:
                logger.warning(f"未找到级联分类器文件: {self.cascade_path}")
        except Exception as e:
            logger.error(f"初始化分类器时出错: {e}")
            self.face_cascade = None
    
    def check_required_models(self):
        """检查必要的模型文件是否存在，并提示用户下载"""
        # 检查dlib特征点预测模型
        if not os.path.exists(self.predictor_path):
            messagebox.showinfo("提示", 
                "首次使用需要下载面部特征点模型。\n"
                "请从 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 下载模型文件，\n"
                "解压后放置到 models 文件夹中。"
            )
        
        # 检查InsightFace模型
        if INSIGHTFACE_AVAILABLE and not os.path.exists(self.inswapper_path):
            self.show_model_download_guide("inswapper")
    
    def show_model_download_guide(self, model_type="inswapper"):
        """显示模型下载指南"""
        if model_type == "inswapper":
            messagebox.showinfo("提示", 
                "需要下载InsightFace人脸替换模型(inswapper_128.onnx)。\n"
                "1. 从https://huggingface.co/deepinsight/inswapper/blob/main/inswapper_128.onnx下载模型\n"
                f"2. 保存到: {self.inswapper_path}\n"
                "3. 确保已安装所需库: pip install insightface onnx onnxruntime"
            )
    
    def create_ui(self):
        # 创建主滚动区域
        self.main_canvas = tk.Canvas(self.root, bg=self.secondary_color, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview, 
                                     style="TScrollbar")
        self.scrollable_frame = ttk.Frame(self.main_canvas, style="TFrame")
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(
                scrollregion=self.main_canvas.bbox("all")
            )
        )
        
        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # 创建应用标题栏
        self.create_title_bar()
        
        # 创建主内容区域
        self.create_main_content()
        
        # 创建高级选项区域
        self.create_advanced_options()
        
        # 创建操作按钮区域
        self.create_action_buttons()
        
        # 进度条和状态区域
        self.create_status_area()
        
        # 视频播放器区域
        self.create_video_player()
        
        # 初始隐藏视频播放器
        self.video_player_frame.pack_forget()
    
    def create_title_bar(self):
        """创建应用标题栏"""
        title_frame = ttk.Frame(self.scrollable_frame, style="TFrame")
        title_frame.pack(fill=tk.X, padx=15, pady=(15, 5))
        
        title_label = ttk.Label(title_frame, text="人脸替换应用 - InsightFace版", 
                               style="Title.TLabel", font=('微软雅黑', 16, 'bold'))
        title_label.pack(pady=5)
        
        subtitle_label = ttk.Label(title_frame, 
                                 text="将视频中的人脸替换为指定的人脸图像", 
                                 style="TLabel")
        subtitle_label.pack(pady=2)
        
        separator = ttk.Separator(self.scrollable_frame, orient='horizontal')
        separator.pack(fill=tk.X, padx=10, pady=(0, 10))
    
    def create_main_content(self):
        """创建主要内容区域"""
        # 文件选择区域
        file_frame = ttk.Frame(self.scrollable_frame, style="TFrame")
        file_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # 使用网格布局提高对齐性
        file_frame.columnconfigure(1, weight=1)  # 让输入框可以伸展
        
        # 视频选择
        ttk.Label(file_frame, text="选择视频:", style="TLabel").grid(row=0, column=0, sticky=tk.W, padx=5, pady=12)
        self.video_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.video_path_var, width=50).grid(row=0, column=1, padx=5, pady=10, sticky=tk.EW)
        ttk.Button(file_frame, text="浏览...", command=self.browse_video).grid(row=0, column=2, padx=5, pady=10)
        ttk.Button(file_frame, text="刷新数据文件夹", command=self.load_data_folder).grid(row=0, column=3, padx=5, pady=10)
        
        # 人脸图片选择
        ttk.Label(file_frame, text="选择人脸图片:", style="TLabel").grid(row=1, column=0, sticky=tk.W, padx=5, pady=12)
        ttk.Button(file_frame, text="浏览...", command=self.browse_faces).grid(row=1, column=1, padx=5, pady=10, sticky=tk.W)
        
        # 输出路径
        ttk.Label(file_frame, text="输出路径:", style="TLabel").grid(row=2, column=0, sticky=tk.W, padx=5, pady=12)
        self.output_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.output_path_var, width=50).grid(row=2, column=1, padx=5, pady=10, sticky=tk.EW)
        ttk.Button(file_frame, text="浏览...", command=self.browse_output).grid(row=2, column=2, padx=5, pady=10)
        
        # 增加说明性提示
        tip_frame = ttk.Frame(self.scrollable_frame, style="TFrame")
        tip_frame.pack(fill=tk.X, padx=25, pady=(0, 10))
        tip_label = ttk.Label(tip_frame, text="提示: 可以从data文件夹中选择视频和图片，或者浏览本地文件", 
                            style="TLabel", foreground="#757575", font=('微软雅黑', 9, 'italic'))
        tip_label.pack(anchor=tk.W)
        
        # 人脸图片显示区域
        ttk.Label(self.scrollable_frame, text="人脸图片:", style="Title.TLabel").pack(anchor=tk.W, padx=25, pady=(10, 5))
        self.face_frame = ttk.Frame(self.scrollable_frame, style="TFrame")
        self.face_frame.pack(fill=tk.X, padx=25, pady=10)
    
    def create_advanced_options(self):
        """创建高级选项区域"""
        # 高级选项标题
        advanced_title = ttk.Label(self.scrollable_frame, text="高级选项", style="Title.TLabel")
        advanced_title.pack(anchor=tk.W, padx=25, pady=(15, 5))
        
        # 高级选项容器
        advanced_frame = ttk.LabelFrame(self.scrollable_frame, text="调整参数", padding=(10, 5))
        advanced_frame.pack(fill=tk.X, padx=25, pady=10)
        
        # 分为左右两列布局
        left_options = ttk.Frame(advanced_frame)
        left_options.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        right_options = ttk.Frame(advanced_frame)
        right_options.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 左侧选项
        # 添加平滑度滑块
        ttk.Label(left_options, text="面部混合平滑度:").pack(anchor=tk.W, pady=(10, 5))
        self.smoothing_var = tk.IntVar(value=50)
        smoothing_scale = ttk.Scale(left_options, variable=self.smoothing_var, from_=0, to=100, 
                                    orient=tk.HORIZONTAL, length=200)
        smoothing_scale.pack(fill=tk.X, pady=5)
        
        smoothing_value_frame = ttk.Frame(left_options)
        smoothing_value_frame.pack(fill=tk.X)
        ttk.Label(smoothing_value_frame, text="低").pack(side=tk.LEFT)
        ttk.Label(smoothing_value_frame, text="高").pack(side=tk.RIGHT)
        
        # 添加颜色校正选项
        color_frame = ttk.Frame(left_options)
        color_frame.pack(fill=tk.X, pady=(15, 5))
        self.color_correction_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(color_frame, text="启用颜色校正", variable=self.color_correction_var).pack(anchor=tk.W)
        
        # 添加多尺度检测选项
        multi_scale_frame = ttk.Frame(left_options)
        multi_scale_frame.pack(fill=tk.X, pady=(15, 5))
        self.multi_scale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(multi_scale_frame, text="启用多尺度人脸检测 (更精确但更慢)", 
                        variable=self.multi_scale_var).pack(anchor=tk.W)
        
        # 右侧选项
        # 添加人脸检测器选择
        ttk.Label(right_options, text="人脸检测器:").pack(anchor=tk.W, pady=(10, 5))
        detector_frame = ttk.Frame(right_options)
        detector_frame.pack(fill=tk.X, pady=5)
        self.detector_var = tk.StringVar(value="dlib")
        ttk.Radiobutton(detector_frame, text="Dlib ", variable=self.detector_var, value="dlib").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(detector_frame, text="OpenCV ", variable=self.detector_var, value="opencv").pack(side=tk.LEFT, padx=5)
        
        # 添加置信度阈值滑块 - 修复resolution参数问题
        ttk.Label(right_options, text="人脸检测置信度阈值:").pack(anchor=tk.W, pady=(15, 5))
        self.confidence_var = tk.DoubleVar(value=0.5)
        # 使用标准tk.Scale而不是ttk.Scale，以支持resolution参数
        confidence_scale = tk.Scale(right_options, variable=self.confidence_var, from_=0.1, to=0.9, resolution=0.1,
                                  orient=tk.HORIZONTAL, length=200, bg=self.secondary_color, 
                                  highlightthickness=0, troughcolor=self.secondary_color,
                                  activebackground=self.primary_color)
        confidence_scale.pack(fill=tk.X, pady=5)
        
        confidence_value_frame = ttk.Frame(right_options)
        confidence_value_frame.pack(fill=tk.X)
        ttk.Label(confidence_value_frame, text="0.1").pack(side=tk.LEFT)
        ttk.Label(confidence_value_frame, text="0.9").pack(side=tk.RIGHT)
        
        # 添加人脸替换方法选择
        ttk.Label(right_options, text="人脸替换方法:").pack(anchor=tk.W, pady=(15, 5))
        swapper_frame = ttk.Frame(right_options)
        swapper_frame.pack(fill=tk.X, pady=5)
        self.swapper_var = tk.StringVar(value="inswapper" if self.inswapper is not None and self.face_analyser is not None else "traditional")
        ttk.Radiobutton(swapper_frame, text="传统方法 ", variable=self.swapper_var, value="traditional").pack(side=tk.LEFT, padx=5)
        
        # 仅当inswapper模型可用时才显示此选项
        if self.inswapper is not None and self.face_analyser is not None:
            ttk.Radiobutton(swapper_frame, text="Inswapper ", variable=self.swapper_var, value="inswapper").pack(side=tk.LEFT, padx=5)
        
    def create_action_buttons(self):
        """创建操作按钮区域"""
        button_frame = ttk.Frame(self.scrollable_frame)
        button_frame.pack(fill=tk.X, padx=25, pady=20)
        
        # 添加预览按钮
        preview_btn = ttk.Button(button_frame, text="预览单帧", command=self.preview_frame,
                              style="TButton", width=15)
        preview_btn.pack(side=tk.LEFT, padx=10)
        
        # 分隔线
        ttk.Separator(button_frame, orient='vertical').pack(side=tk.LEFT, fill='y', padx=20, pady=10)
        
        # 开始处理按钮
        start_btn = ttk.Button(button_frame, text="开始处理", command=self.start_processing,
                            style="Success.TButton", width=20)
        start_btn.pack(side=tk.RIGHT, padx=10)
    
    def create_status_area(self):
        """创建状态和进度区域"""
        status_frame = ttk.Frame(self.scrollable_frame)
        status_frame.pack(fill=tk.X, padx=25, pady=10)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                         style="Horizontal.TProgressbar",
                                         orient="horizontal", length=500, mode="determinate")
        self.progress_bar.pack(pady=10, fill=tk.X)
        
        # 进度百分比标签
        self.progress_label = ttk.Label(status_frame, text="0%", style="TLabel")
        self.progress_label.pack(pady=5)
        
        # 状态标签
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                              style="TLabel", font=('微软雅黑', 11))
        status_label.pack(pady=5)
    
    def create_video_player(self):
        """创建视频播放器区域"""
        self.video_player_frame = ttk.LabelFrame(self.scrollable_frame, text="视频播放器", padding=10)
        self.video_player_frame.pack(fill=tk.X, padx=25, pady=15)
        
        # 视频显示区域
        self.video_canvas = tk.Canvas(self.video_player_frame, bg="black", width=640, height=360, 
                                    highlightthickness=0)
        self.video_canvas.pack(pady=10)
        
        # 播放控制区
        controls_frame = ttk.Frame(self.video_player_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        # 播放/暂停按钮
        self.play_button = ttk.Button(controls_frame, text="▶ 播放", command=self.toggle_play, width=10)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        # 停止按钮
        ttk.Button(controls_frame, text="■ 停止", command=self.stop_video, width=10).pack(side=tk.LEFT, padx=5)
        
        # 视频进度条
        self.video_progress_var = tk.DoubleVar(value=0)
        self.video_progress = ttk.Scale(controls_frame, variable=self.video_progress_var, 
                                      from_=0, to=100, orient=tk.HORIZONTAL, length=400,
                                      command=self.seek_video)
        self.video_progress.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # 视频时间标签
        self.time_label = ttk.Label(controls_frame, text="00:00 / 00:00", style="TLabel")
        self.time_label.pack(side=tk.LEFT, padx=5)
    
    def load_data_folder(self):
        """自动加载data文件夹中的视频和图片"""
        self.status_var.set("正在加载数据文件夹...")
        
        # 加载视频
        video_files = []
        for file in os.listdir(self.data_folder):
            if file.lower().endswith(('.mp4')):
                video_files.append(os.path.join(self.data_folder, file))
        
        if video_files:
            self.video_path = video_files[0]  # 选择第一个视频
            self.video_path_var.set(self.video_path)
        
        # 加载图片
        image_files = []
        for file in os.listdir(self.data_folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(self.data_folder, file))
        
        if image_files:
            self.face_images = image_files
            self.display_face_images()
            
            # 自动生成输出路径
            if self.video_path:
                video_name = os.path.splitext(os.path.basename(self.video_path))[0]
                self.output_path = os.path.join(self.output_folder, f"{video_name}_face_swap.mp4")
                self.output_path_var.set(self.output_path)
        
        self.status_var.set(f"已加载 {len(video_files)} 个视频和 {len(image_files)} 张图片")
    
    def browse_video(self):
        """浏览选择视频文件"""
        # 默认打开data文件夹
        initial_dir = self.data_folder if os.path.exists(self.data_folder) else "/"
        path = filedialog.askopenfilename(initialdir=initial_dir,
                                         title="选择视频文件",
                                         filetypes=(("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")))
        if path:
            self.video_path = path
            self.video_path_var.set(path)
            self.status_var.set(f"已选择视频: {os.path.basename(path)}")
    
    def browse_faces(self):
        """浏览选择人脸图片"""
        # 默认打开data文件夹
        initial_dir = self.data_folder if os.path.exists(self.data_folder) else "/"
        paths = filedialog.askopenfilenames(initialdir=initial_dir,
                                          title="选择人脸图片",
                                          filetypes=(("图片文件", "*.jpg *.jpeg *.png"), ("所有文件", "*.*")))
        if paths:
            self.face_images = list(paths)
            self.display_face_images()
            self.status_var.set(f"已选择 {len(paths)} 张人脸图片")
    
    def browse_output(self):
        """浏览选择输出文件路径"""
        # 默认打开output_videos文件夹
        initial_dir = self.output_folder if os.path.exists(self.output_folder) else "/"
        filename = os.path.basename(self.video_path) if self.video_path else "output.mp4"
        filename = "output_" + filename if not filename.startswith("output_") else filename
        path = filedialog.asksaveasfilename(initialdir=initial_dir,
                                          initialfile=filename,
                                          title="保存输出视频",
                                          filetypes=(("MP4文件", "*.mp4"), ("所有文件", "*.*")))
        if path:
            # 确保有.mp4后缀
            if not path.lower().endswith('.mp4'):
                path += '.mp4'
            self.output_path = path
            self.output_path_var.set(path)
            self.status_var.set(f"输出将保存至: {os.path.basename(path)}")
    
    def display_face_images(self):
        """显示所有人脸图片"""
        # 清除现有的图片
        for widget in self.face_frame.winfo_children():
            widget.destroy()
        
        # 如果没有图片，显示提示
        if not self.face_images:
            ttk.Label(self.face_frame, text="未选择人脸图片", style="TLabel").pack(pady=20)
            return
        
        # 显示所有人脸图片
        for i, img_path in enumerate(self.face_images):
            # 创建带边框的框架
            image_frame = ttk.Frame(self.face_frame, padding=5)
            image_frame.grid(row=0, column=i, padx=10, pady=10)
            
            try:
            # 加载并调整图片大小
            img = Image.open(img_path)
            img = img.resize((150, 150), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # 保存引用以防止垃圾回收
                image_frame.photo = photo
            
            # 显示图片
                img_label = ttk.Label(image_frame, image=photo)
                img_label.pack(padx=5, pady=5)
                
                # 图片名称标签
                filename = os.path.basename(img_path)
                if len(filename) > 20:
                    filename = filename[:17] + "..."
                name_label = ttk.Label(image_frame, text=filename, style="TLabel")
                name_label.pack(pady=(0, 5))
            
            # 添加选择按钮
                select_btn = ttk.Button(image_frame, text=f"选择图片 {i+1}", 
                                     command=lambda idx=i: self.select_face(idx),
                                     style="TButton", width=15)
                select_btn.pack(pady=5)
            except Exception as e:
                # 处理图片加载失败的情况
                error_frame = ttk.Frame(self.face_frame, padding=5)
                error_frame.grid(row=0, column=i, padx=10, pady=10)
                
                error_label = ttk.Label(error_frame, text="图片加载失败", style="TLabel")
                error_label.pack(pady=20)
                
                ttk.Label(error_frame, text=f"错误: {str(e)[:30]}", style="TLabel", 
                        foreground="red").pack(pady=5)
    
    def select_face(self, index):
        """选择人脸图片"""
        self.selected_face_index = index
        
        # 更新UI以显示选择
        for i, frame in enumerate(self.face_frame.winfo_children()):
            if i == index:
                # 创建选中效果
                frame.configure(relief="raised", borderwidth=2)
                for child in frame.winfo_children():
                    if isinstance(child, ttk.Button):
                        child.configure(text="✓ 已选择")
            else:
                # 恢复未选中状态
                frame.configure(relief="flat", borderwidth=0)
                for child in frame.winfo_children():
                    if isinstance(child, ttk.Button):
                        child.configure(text=f"选择图片 {i+1}")
    
    def start_processing(self):
        if not self.video_path:
            messagebox.showerror("错误", "请选择输入视频")
            return
        
        if not self.face_images:
            messagebox.showerror("错误", "请选择至少一张人脸图片")
            return
        
        if self.selected_face_index < 0:
            messagebox.showerror("错误", "请选择一张要替换的人脸图片")
            return
        
        if not self.output_path:
            messagebox.showerror("错误", "请指定输出路径")
            return
        
        # 获取用户选择的人脸替换方法
        swapper_choice = getattr(self, 'swapper_var', tk.StringVar(value="traditional")).get()
        
        # 如果选择了inswapper但模型未加载，提示下载或回退到传统方法
        if swapper_choice == "inswapper":
            if self.inswapper is None or self.face_analyser is None:
                response = messagebox.askquestion("模型缺失", 
                                               "缺少Inswapper模型或Face Analyser。\n是否使用传统替换方法继续？\n\n点击'否'将取消处理。")
                if response == 'yes':
                    swapper_choice = "traditional"
                else:
                    return
        
        # 如果使用传统方法，检查是否有特征点预测器
        if swapper_choice == "traditional":
            predictor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "shape_predictor_68_face_landmarks.dat")
            if not os.path.exists(predictor_path):
                messagebox.showerror("错误", "缺少面部特征点模型文件。请从 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 下载并解压到 models 文件夹。")
                return
            
            if self.predictor is None:
                self.predictor = dlib.shape_predictor(predictor_path)
        
        # 在新线程中启动处理，以避免UI冻结
        threading.Thread(target=self.process_video, daemon=True).start()
    
    def insightface_face_swap(self, frame, source_img):
        """使用InsightFace进行人脸替换"""
        try:
            # 检查人脸分析器和交换器
            if self.face_analyser is None or self.inswapper is None:
                return frame  # 直接返回原始帧
            
            # 检查输入帧是否有效
            if frame is None or frame.size == 0:
                    return frame
            
            # 将BGR帧转换为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 检测和分析人脸
            input_faces = self.face_analyser.get(rgb_frame)
            if len(input_faces) == 0:
                return frame  # 没有检测到人脸
                
            # 分析源图像中的人脸
            source_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
            source_faces = self.face_analyser.get(source_rgb)
                if len(source_faces) == 0:
                return frame  # 源图像中没有检测到人脸
            
            # 获取源人脸
            source_face = source_faces[0]
            
            # 替换每个检测到的人脸
            result = frame.copy()
            
            # 检查是否需要颜色校正
            use_color_correction = False
            if hasattr(self, 'color_correction_var'):
                if hasattr(self.color_correction_var, 'get'):
                    # 如果是Tkinter变量
                    use_color_correction = self.color_correction_var.get()
                else:
                    # 如果是普通变量（布尔值）
                    use_color_correction = self.color_correction_var
            
            for face in input_faces:
                # 应用人脸交换
                result = self.inswapper.get(result, face, source_face, paste_back=True)
                
                # 应用颜色校正（如果启用）
                if use_color_correction:
                    # 创建面部区域蒙版
                    mask = np.zeros_like(frame)
                    
                    # 获取人脸关键点
                    kps = face.kps.astype(int)
                    hull = cv2.convexHull(kps)
                    
                    # 绘制凸包形成蒙版
                    cv2.fillConvexPoly(mask, hull, (255, 255, 255))
                    
                    # 应用颜色校正
                    result = self.adjust_color_balance(result, frame, mask=mask[:,:,0], blur_amount=10)
            
            return result
            
        except Exception as e:
            logger.error(f"InsightFace换脸过程出错: {e}")
            import traceback
            traceback.print_exc()
            return frame  # 出错时返回原始帧
    
    def adjust_color_balance(self, target_img, source_img, mask=None, blur_amount=0):
        """
        调整目标图像的颜色平衡以匹配源图像
        这是一个简化版的颜色校正，适用于InsightFace处理后的结果
        """
        try:
            # 转换为LAB色彩空间
            target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB)
            source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)
            
            # 分离通道
            target_l, target_a, target_b = cv2.split(target_lab)
            source_l, source_a, source_b = cv2.split(source_lab)
            
            # 计算源图像和目标图像的均值和标准差
            source_l_mean, source_l_std = cv2.meanStdDev(source_l)
            target_l_mean, target_l_std = cv2.meanStdDev(target_l)
            
            # 调整亮度通道
            l_adjusted = target_l * (source_l_std[0][0] / target_l_std[0][0]) + (source_l_mean[0][0] - target_l_mean[0][0] * (source_l_std[0][0] / target_l_std[0][0]))
            
            # 确保值在有效范围内
            l_adjusted = np.clip(l_adjusted, 0, 255).astype(np.uint8)
            
            # 合并通道
            adjusted_lab = cv2.merge([l_adjusted, target_a, target_b])
            
            # 转换回BGR色彩空间
            adjusted_img = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)
            
            # 使用alpha混合平滑过渡
            alpha = 0.7  # 混合因子
            result = cv2.addWeighted(adjusted_img, alpha, target_img, 1-alpha, 0)
            
            # 应用高斯模糊（如果启用）
            if blur_amount > 0:
                result = cv2.GaussianBlur(result, (blur_amount, blur_amount), 0)
            
            return result
            
        except Exception as e:
            logger.error(f"颜色平衡调整出错: {e}")
            return target_img
    
    def process_video(self):
        """处理视频的主函数，应用人脸替换效果并保存结果"""
        try:
            # 检查是否选择了视频
            if not self.video_path:
                # 使用回调或UI更新状态
                self.update_status("请先选择一个视频文件")
                return False
            
            # 检查视频文件是否存在
            if not os.path.exists(self.video_path):
                self.update_status(f"视频文件不存在: {self.video_path}")
                return False
            
            # 检查是否选择了人脸图片
            if not self.face_images or self.selected_face_index < 0 or self.selected_face_index >= len(self.face_images):
                self.update_status("请先选择一张人脸图片")
                return False
            
            # 检查人脸图片是否存在
            target_face_path = self.face_images[self.selected_face_index]
            if not os.path.exists(target_face_path):
                self.update_status(f"人脸图片不存在: {target_face_path}")
                return False
            
            # 检查是否设置了输出路径
            if not self.output_path:
                # 根据输入文件创建一个默认输出路径
                video_dir = os.path.dirname(self.video_path)
                video_name = os.path.basename(self.video_path)
                base_name, ext = os.path.splitext(video_name)
                self.output_path = os.path.join(video_dir, f"{base_name}_face_swap{ext}")
            
            # 更新状态
            self.update_status("正在处理视频...")
            
            # 初始化日志文件
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            # 打开视频
            video = cv2.VideoCapture(self.video_path)
            if not video.isOpened():
                self.update_status(f"无法打开视频文件: {self.video_path}")
                return False
            
            # 获取视频基本信息
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 保存视频总帧数，供其他函数使用
            self.total_frames = total_frames
            
            # 创建目录
            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
            output_file = self.output_path
            
            # 检查文件是否已存在，如果存在则添加时间戳
            if os.path.exists(output_file):
                base_name, ext = os.path.splitext(output_file)
                output_file = f"{base_name}_{int(time.time())}{ext}"
                self.output_path = output_file
            
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            
            # 更新状态
            logger.info(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
            logger.info(f"输出文件: {output_file}")
            self.update_status(f"正在处理视频: {width}x{height}, {fps}fps, {total_frames}帧")
            
            # 使用线程池并行处理视频帧
            max_workers = os.cpu_count() or 4  # 如果无法获取CPU核心数，则默认使用4个线程
            
            # 如果使用InsightFace，由于GPU内存限制，减少工作线程数量
            if self.swapper_var == "inswapper" and self.inswapper is not None:
                # GPU模式下只使用1个线程，否则模型会占用大量显存
                max_workers = min(2, max_workers)
                logger.info(f"使用InsightFace，限制工作线程数为: {max_workers}")
            
            # 更新进度
            self.update_progress(0, "准备处理")
            
            # 使用线程池处理视频帧
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 保存所有处理任务的future对象
                futures = set()
                
                # 处理一帧视频的任务
                def process_frame_task(frame_data):
                    # 解构参数
                    index, frame = frame_data
                    
                    # 获取用户选择的人脸替换方法和参数
                    swapper_choice = self.swapper_var
                    detector_choice = self.detector_var
                    target_face_path = self.face_images[self.selected_face_index]
                
                    # 处理图像 - 这里需要适配你的处理流程
                    processed_frame = None
                    
                    # 根据不同的处理方法调用不同的函数
                    if swapper_choice == "inswapper" and self.inswapper is not None and self.face_analyser is not None:
                        # 读取目标人脸
                        target_image = cv2.imread(target_face_path)
                        if target_image is None:
                            raise ValueError(f"无法读取人脸图片: {target_face_path}")
                        
                        # 使用InsightFace替换
                        processed_frame = self.insightface_face_swap(frame, target_image)
                        
                        else:
                        # 传统方法处理
                        use_multi_scale = self.multi_scale_var
                        target_image_rgb = cv2.imread(target_face_path)
                        if target_image_rgb is None:
                            raise ValueError(f"无法读取人脸图片: {target_face_path}")
                            
                        # 处理目标图像的人脸特征点
                        target_landmarks = None
                        
                        # 检测目标人脸
                        target_rgb = cv2.cvtColor(target_image_rgb, cv2.COLOR_BGR2RGB)
                        if detector_choice == "dlib":
                            target_faces = self.detector(target_rgb, 1)
                            if len(target_faces) > 0:
                                target_landmarks = self.predictor(target_rgb, target_faces[0])
                                target_landmarks = self.shape_to_np(target_landmarks)
                        else:  # OpenCV检测器
                            target_gray = cv2.cvtColor(target_image_rgb, cv2.COLOR_BGR2GRAY)
                            target_faces = self.opencv_detector.detectMultiScale(
                                target_gray, 
                                scaleFactor=1.1, 
                                minNeighbors=5,
                                minSize=(30, 30)
                            )
                            
                            if len(target_faces) > 0:
                                x, y, w, h = target_faces[0]
                                target_face_rect = dlib.rectangle(x, y, x+w, y+h)
                                target_landmarks = self.predictor(target_rgb, target_face_rect)
                                target_landmarks = self.shape_to_np(target_landmarks)
                        
                        if target_landmarks is None:
                            # 如果没有检测到目标人脸，返回原始帧
                            return (index, frame)
                        
                        # 使用传统方法处理帧
                        processed_frame = self.process_frame_traditional(
                            frame, 
                            target_image_rgb, 
                            target_landmarks, 
                            detector_choice, 
                            use_multi_scale
                        )
                    
                    # 返回处理后的帧和索引
                    if processed_frame is not None:
                        return (index, processed_frame)
                    else:
                        # 如果处理失败，返回原始帧
                        return (index, frame)
                
                # 初始化变量
                frame_count = 0
                frame_buffer = {}  # 存储处理好的帧
                next_frame_to_write = 0  # 下一个要写入的帧索引
                
                while True:
                    ret, frame = video.read()
                    if not ret:
                        break
                    
                    # 更新进度
                    frame_count += 1
                    progress = (frame_count / total_frames) * 100
                    # 使用回调更新进度
                    self.update_progress(progress, f"处理帧 {frame_count}/{total_frames}")
                    
                    # 每100帧显示一次进度
                    if frame_count % 100 == 0 or frame_count == total_frames:
                        self.update_status(f"正在处理视频... {frame_count}/{total_frames} 帧")
                        logger.info(f"处理进度: {frame_count}/{total_frames} ({progress:.1f}%)")
                    
                    # 提交处理任务
                    future = executor.submit(process_frame_task, (frame_count-1, frame))
                    futures.add(future)
                    
                    # 检查已完成的任务
                    done_futures = set()
                    for future in list(futures):
                        if future.done():
                            try:
                                # 获取处理结果
                                idx, processed_frame = future.result()
                                # 将结果添加到帧缓冲区
                                frame_buffer[idx] = processed_frame
                                
                                # 从futures集合中移除已完成的任务
                                done_futures.add(future)
                                
                                # 检查是否可以按顺序写入帧
                                if idx == next_frame_to_write:
                                    next_frame_to_write = self.write_frames_in_order(out, frame_buffer, next_frame_to_write)
                                
                            except Exception as e:
                                logger.error(f"处理帧时出错: {e}")
                                import traceback
                                traceback.print_exc()
                    
                    # 从futures集合中移除已完成的任务
                    futures -= done_futures
                    
                    # 控制任务队列大小，避免内存溢出
                    max_pending_tasks = max_workers * 3  # 允许的最大待处理任务数
                    while len(futures) > max_pending_tasks:
                        # 等待其中一个任务完成
                        done, futures_set = concurrent.futures.wait(
                            futures, 
                            return_when=concurrent.futures.FIRST_COMPLETED,
                            timeout=1.0  # 设置超时，防止无限等待
                        )
                        
                        # 处理完成的任务
                        for future in done:
                            try:
                                idx, processed_frame = future.result()
                                frame_buffer[idx] = processed_frame
                                
                                # 检查是否可以按顺序写入帧
                                if idx == next_frame_to_write:
                                    next_frame_to_write = self.write_frames_in_order(out, frame_buffer, next_frame_to_write)
                                
                            except Exception as e:
                                logger.error(f"处理帧时出错: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        # 更新futures集合
                        futures = futures_set
                
                # 等待所有任务完成
                for future in concurrent.futures.as_completed(futures):
                    try:
                        idx, processed_frame = future.result()
                        frame_buffer[idx] = processed_frame
                    except Exception as e:
                        logger.error(f"处理帧时出错: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 写入剩余的帧
                while len(frame_buffer) > 0:
                    next_frame_to_write = self.write_frames_in_order(out, frame_buffer, next_frame_to_write)
                    # 如果没有更多帧可写，则退出循环
                    if next_frame_to_write >= total_frames:
                        break
                
                # 释放资源
                video.release()
                out.release()
                
            # 更新状态
            self.update_status("处理完成!")
            self.update_progress(100, "完成")
            
            logger.info(f"视频处理完成: {self.output_path}")
            return self.output_path
            
        except Exception as e:
            self.update_status(f"错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def update_progress(self, value, text=None):
        """更新进度条，支持无UI模式"""
        try:
            # 如果有progress_var（Tkinter界面）
            if hasattr(self, 'progress_var') and self.progress_var is not None:
                self.progress_var.set(value)
                # 如果有progress_label
                if hasattr(self, 'progress_label') and self.progress_label is not None and self.root is not None:
                    progress_text = f"{value:.1f}%"
                    self.root.after(0, lambda t=progress_text: self.progress_label.config(text=t))
            # 打印日志
            logger.info(f"进度: {value:.1f}% {text if text else ''}")
        except Exception as e:
            logger.error(f"更新进度时出错: {e}")
            
    def update_status(self, text):
        """更新状态文本，支持无UI模式"""
        try:
            # 如果有status_var（Tkinter界面）
            if hasattr(self, 'status_var') and self.status_var is not None:
                self.status_var.set(text)
            # 打印日志
            logger.info(f"状态: {text}")
        except Exception as e:
            logger.error(f"更新状态时出错: {e}")
            
    def write_frames_in_order(self, out, frames_buffer, next_frame_to_write):
        """按顺序写入已处理的视频帧，返回下一个要写入的帧索引"""
        # 复制一份帧缓冲区的键，并进行排序
        keys = sorted(list(frames_buffer.keys()))
        
        # 确保有需要写入的帧
        if not keys:
            return next_frame_to_write
        
        # 找出连续的帧并写入
        while keys and keys[0] == next_frame_to_write:
            # 写入帧并从缓冲区中移除
            frame = frames_buffer.pop(keys[0])
                        
            # 确保帧是有效的
            if frame is not None and frame.size > 0:
                try:
                    out.write(frame)
                except Exception as e:
                    logger.error(f"写入帧 {next_frame_to_write} 时出错: {e}")
                        else:
                logger.error(f"帧 {next_frame_to_write} 无效，跳过")
            
            # 更新下一个要写入的帧索引
            next_frame_to_write += 1
            
            # 从排序后的键列表中移除已处理的键
            keys.pop(0)
        
        # 更新进度条
        if hasattr(self, 'progress_var'):
            # 安全获取total_frames属性
            total_frames = getattr(self, 'total_frames', 0)
            if total_frames > 0:
                progress_value = min(100, int((next_frame_to_write / total_frames) * 100))
                self.progress_var.set(progress_value)
                # 更新进度标签
                progress_text = f"{progress_value}%"
                self.root.after(0, lambda t=progress_text: self.progress_label.config(text=t))
                
                # 更新状态文本
                status_text = f"处理中... {progress_value}%"
                self.root.after(0, lambda t=status_text: self.status_var.set(t))
                
        return next_frame_to_write

    def load_video_player(self, video_path):
        """加载视频到播放器"""
        try:
            # 先初始化视频播放相关变量，防止stop_video方法中访问未定义的属性
            self.duration = 0
            self.fps = 0
            self.total_frames = 0
            self.current_frame = 0
            self.is_playing = False
            self.play_thread = None
            
            # 显示视频播放器框架
            self.video_player_frame.pack(fill=tk.X, padx=10, pady=10)
                        
            # 获取视频信息
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                messagebox.showerror("错误", f"无法打开视频: {video_path}")
                return
            
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration = self.total_frames / self.fps if self.fps > 0 else 0
            
            # 设置视频尺寸适应画布
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 设置视频实际播放尺寸
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
                    
            # 设置缩放比例，保持宽高比
            ratio = min(canvas_width/width, canvas_height/height)
            self.video_width = int(width * ratio)
            self.video_height = int(height * ratio)
            
            # 存储视频对象
            self.cap = cap
            
            # 更新状态
            total_time = self.format_time(self.duration)
            self.time_label.config(text=f"00:00 / {total_time}")
            self.status_var.set(f"视频已加载: {os.path.basename(video_path)} ({width}x{height}, {self.fps:.1f}fps)")
                    
            # 显示第一帧
            self.update_video_frame(0)
            
            # 显示成功提示
            messagebox.showinfo("处理完成", "视频处理完成，可以在播放器中查看结果")
                except Exception as e:
            logger.error(f"加载视频失败: {str(e)}")
            messagebox.showerror("错误", f"加载视频失败: {str(e)}")
            self.stop_video()
    
    def stop_video(self):
        """停止视频播放"""
        self.is_playing = False
        if hasattr(self, 'play_button'):  # 确保按钮已经创建
            self.play_button.config(text="▶ 播放")
        self.play_thread = None
        self.current_frame = 0
        if hasattr(self, 'video_progress_var'):  # 确保进度条已经创建
            self.video_progress_var.set(0)
        
        # 确保相关属性和组件都存在
        if hasattr(self, 'time_label') and hasattr(self, 'duration'):
            # 处理duration可能不存在的情况
            duration = getattr(self, 'duration', 0)
            if hasattr(self, 'format_time'):
                self.time_label.config(text=f"00:00 / {self.format_time(duration)}")
                    else:
                # 如果format_time方法不存在，使用简单格式
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                self.time_label.config(text=f"00:00 / {minutes:02d}:{seconds:02d}")
        
        # 显示第一帧（如果有）
        if hasattr(self, 'video_frames') and self.video_frames and hasattr(self, 'video_canvas'):
            self.video_canvas.delete("video_frame")
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.video_frames[0][1], tags="video_frame")
    
    def toggle_play(self):
        """切换视频播放/暂停状态"""
        if not hasattr(self, 'cap') or self.cap is None:
            return
            
        if self.is_playing:
            # 暂停播放
            self.is_playing = False
            self.play_button.config(text="▶ 播放")
        else:
            # 开始/继续播放
            self.is_playing = True
            self.play_button.config(text="⏸ 暂停")
            
            # 启动播放线程
            if self.play_thread is None or not self.play_thread.is_alive():
                self.play_thread = threading.Thread(target=self.play_video)
                self.play_thread.daemon = True
                self.play_thread.start()
    
    def play_video(self):
        """在单独线程中播放视频"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return
            
            # 跳转到当前帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            
            while self.is_playing and self.play_thread:
                # 检查是否应当停止
                if not self.is_playing or self.current_frame >= self.total_frames - 1:
                    break
                
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 更新当前帧
                self.current_frame += 1
                
                # 更新进度条和时间
                progress = (self.current_frame / self.total_frames) * 100
                self.video_progress_var.set(progress)
                
                current_time = self.current_frame / self.fps if self.fps > 0 else 0
                current_time_str = self.format_time(current_time)
                duration_str = self.format_time(self.duration)
                
                # 使用主线程更新UI
                self.root.after(0, lambda: self.time_label.config(text=f"{current_time_str} / {duration_str}"))
                
                # 转换并显示帧
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                canvas_width = self.video_canvas.winfo_width()
                canvas_height = self.video_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    pil_img = self.resize_image_aspect_ratio(pil_img, min(canvas_height, 360))
                
                img_tk = ImageTk.PhotoImage(image=pil_img)
                
                # 在主线程中更新画布
                self.root.after(0, lambda: self.update_video_frame(img_tk))
                
                # 控制播放速度
                time.sleep(1/self.fps)
            
            # 播放结束，更新按钮状态
            if self.current_frame >= self.total_frames - 1:
                self.root.after(0, lambda: self.play_button.config(text="▶ 播放"))
                self.is_playing = False
            
            cap.release()

        except Exception as e:
            logger.error(f"播放视频时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_video_frame(self, img_tk):
        """更新视频帧，在主线程中调用"""
        self.video_canvas.delete("video_frame")
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk, tags="video_frame")
        # 保持引用以防止垃圾回收
        self.current_img_tk = img_tk
    
    def seek_video(self, value):
        """拖动进度条时调用"""
        if not self.video_path or not os.path.exists(self.video_path):
            return
        
        # 暂停当前播放
        was_playing = self.is_playing
        self.is_playing = False
        self.play_thread = None
        
        # 计算新的帧位置
        progress = float(value)
        new_frame = int((progress / 100) * self.total_frames)
        
        # 防止越界
        new_frame = max(0, min(new_frame, self.total_frames - 1))
        
        # 更新当前帧位置
        self.current_frame = new_frame
        
        # 显示对应帧
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        ret, frame = cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                pil_img = self.resize_image_aspect_ratio(pil_img, min(canvas_height, 360))
            
            img_tk = ImageTk.PhotoImage(image=pil_img)
            self.update_video_frame(img_tk)
            
            # 更新时间显示
            current_time = new_frame / self.fps if self.fps > 0 else 0
            current_time_str = self.format_time(current_time)
            duration_str = self.format_time(self.duration)
            self.time_label.config(text=f"{current_time_str} / {duration_str}")
        
        cap.release()
        
        # 如果之前在播放，继续播放
        if was_playing:
            self.toggle_play()
    
    def format_time(self, seconds):
        """将秒数格式化为分:秒格式"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def advanced_face_swap(self, frame, rgb_frame, landmarks, target_image, target_landmarks):
        """使用改进的方法进行人脸替换"""
        try:
            # 确保所有特征点都是正确的数据类型
            landmarks = np.array(landmarks, dtype=np.int32)
            target_landmarks = np.array(target_landmarks, dtype=np.int32)
            
            # 计算凸包 - 使用完整的68点凸包计算
            hull = cv2.convexHull(landmarks)
            target_hull = cv2.convexHull(target_landmarks)
            
            # 创建掩码 - 增加调试输出
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            
            # 调试掩码
            mask_sum = np.sum(mask)
            print(f"原始掩码像素和: {mask_sum}, 是否为零: {mask_sum == 0}")
            
            # 如果掩码全为零，尝试使用更简单的方法创建掩码
            if mask_sum == 0:
                print("尝试使用备选方法创建掩码")
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                x, y, w, h = cv2.boundingRect(landmarks)
                # 使用椭圆填充而不是凸多边形
                center = (x + w//2, y + h//2)
                axes = (w//2, h//2)
                cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                mask_sum = np.sum(mask)
                print(f"备选掩码像素和: {mask_sum}")
                
                # 保存调试图像
                debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug")
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, "face_mask_debug.jpg"), mask)
            
            # 计算德劳内三角剖分
            rect = cv2.boundingRect(hull)
            subdiv = cv2.Subdiv2D(rect)
            
            # 确保点的格式正确
            points = []
            for point in landmarks:
                points.append((int(point[0]), int(point[1])))
            
            # 插入点到三角剖分
            for point in points:
                try:
                    subdiv.insert(point)
                except Exception as e:
                    print(f"插入点到三角剖分出错: {e}, 点: {point}")
                    continue
            
            # 获取三角形列表
            try:
                triangles = subdiv.getTriangleList()
                triangles = np.array(triangles, dtype=np.int32)
            except Exception as e:
                print(f"获取三角形列表出错: {e}")
                # 如果三角剖分失败，尝试使用简化方法
                return self.simple_face_swap(frame, rgb_frame, landmarks, target_image, target_landmarks)
            
            # 转换三角形索引
            triangle_indexes = []
            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])
                
                idx1 = self.find_point_index(points, pt1)
                idx2 = self.find_point_index(points, pt2)
                idx3 = self.find_point_index(points, pt3)
                
                if idx1 != -1 and idx2 != -1 and idx3 != -1:
                    triangle_indexes.append([idx1, idx2, idx3])
            
            # 对每个三角形进行变换
            result_img = frame.copy()
            
            # 输出三角形数量，帮助调试
            print(f"检测到 {len(triangle_indexes)} 个三角形")
            
            # 如果三角形数量过少，可能是人脸检测不准确
            if len(triangle_indexes) < 10:
                print("警告：三角形数量过少，可能是人脸检测不准确")
                # 保存一个调试图像，显示特征点
                debug_frame = frame.copy()
                for point in landmarks:
                    cv2.circle(debug_frame, (point[0], point[1]), 2, (0, 255, 0), -1)
                debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug")
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, "frame_landmarks_debug.jpg"), debug_frame)
                
                # 尝试使用更简单的方法进行人脸替换
                print("尝试使用备选方法进行人脸替换")
                return self.simple_face_swap(frame, rgb_frame, landmarks, target_image, target_landmarks)
            
            # 继续处理三角形
            for triangle in triangle_indexes:
                try:
                    # 获取三角形顶点
                    tr1_pt1 = landmarks[triangle[0]]
                    tr1_pt2 = landmarks[triangle[1]]
                    tr1_pt3 = landmarks[triangle[2]]
                    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], dtype=np.int32)
                    
                    # 计算三角形的边界矩形
                    rect1 = cv2.boundingRect(triangle1)
                    (x, y, w, h) = rect1
                    
                    # 确保边界在图像范围内
                    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                        continue
                    
                    # 创建裁剪后的三角形
                    cropped_triangle = frame[y:y+h, x:x+w].copy()
                    cropped_mask = np.zeros((h, w), dtype=np.uint8)
                    
                    # 调整坐标
                    points = np.array([[tr1_pt1[0]-x, tr1_pt1[1]-y],
                                      [tr1_pt2[0]-x, tr1_pt2[1]-y],
                                      [tr1_pt3[0]-x, tr1_pt3[1]-y]], dtype=np.int32)
                    
                    cv2.fillConvexPoly(cropped_mask, points, 255)
                    
                    # 获取目标图像中对应的三角形
                    tr2_pt1 = target_landmarks[triangle[0]]
                    tr2_pt2 = target_landmarks[triangle[1]]
                    tr2_pt3 = target_landmarks[triangle[2]]
                    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], dtype=np.int32)
                    
                    # 计算目标三角形的边界矩形
                    rect2 = cv2.boundingRect(triangle2)
                    (x2, y2, w2, h2) = rect2
                    
                    # 确保边界在目标图像范围内
                    if x2 < 0 or y2 < 0 or x2 + w2 > target_image.shape[1] or y2 + h2 > target_image.shape[0]:
                        continue
                    
                    # 创建裁剪后的目标三角形
                    cropped_target = target_image[y2:y2+h2, x2:x2+w2].copy()
                    cropped_target_mask = np.zeros((h2, w2), dtype=np.uint8)
                    
                    # 调整坐标
                    points2 = np.array([[tr2_pt1[0]-x2, tr2_pt1[1]-y2],
                                       [tr2_pt2[0]-x2, tr2_pt2[1]-y2],
                                       [tr2_pt3[0]-x2, tr2_pt3[1]-y2]], dtype=np.int32)
                    
                    cv2.fillConvexPoly(cropped_target_mask, points2, 255)
                    
                    # 计算仿射变换
                    points = np.float32(points)
                    points2 = np.float32(points2)
                    
                    # 确保有足够的点进行变换
                    if len(points) < 3 or len(points2) < 3:
                        continue
                    
                    M = cv2.getAffineTransform(points2, points)
                    
                    # 应用仿射变换
                    warped_triangle = cv2.warpAffine(cropped_target, M, (w, h))
                    
                    # 确保掩码和图像大小匹配
                    if cropped_mask.shape[:2] != warped_triangle.shape[:2]:
                        continue
                    
                    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_mask)
                    
                    # 重建目标图像区域
                    img_area = result_img[y:y+h, x:x+w].copy()
                    
                    # 创建反掩码
                    inv_cropped_mask = cv2.bitwise_not(cropped_mask)
                    
                    # 使用反掩码清除目标区域中将被替换的部分
                    img_area_masked = cv2.bitwise_and(img_area, img_area, mask=inv_cropped_mask)
                    
                    # 将变形后的三角形添加到目标区域
                    img_area = cv2.add(img_area_masked, warped_triangle)
                    
                    # 更新结果图像
                    result_img[y:y+h, x:x+w] = img_area
                except Exception as e:
                    print(f"处理三角形时出错: {e}")
                    continue
            
            # 应用平滑度设置
            smoothing_factor = self.smoothing_var.get() / 10.0  # 将滑块值转换为合适的平滑因子
            
            # 创建无缝克隆的掩码
            center_face = (rect[0] + rect[2]//2, rect[1] + rect[3]//2)
            
            # 添加调试信息
            print(f"人脸中心点: {center_face}, 图像尺寸: {frame.shape[:2]}")
            
            # 确保掩码和图像大小匹配
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            
            # 确保中心点在图像内部
            if 0 <= center_face[0] < frame.shape[1] and 0 <= center_face[1] < frame.shape[0]:
                try:
                    # 确保掩码和图像大小完全匹配
                    if mask.shape[:2] != frame.shape[:2]:
                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    
                    # 检查掩码是否有效（至少有一些非零像素）
                    mask_sum = np.sum(mask)
                    print(f"最终掩码像素和: {mask_sum}")
                    
                    if mask_sum > 0:
                        # 计算掩码的边界框，确保在图像内部
                        mask_indices = np.where(mask > 0)
                        if len(mask_indices[0]) > 0 and len(mask_indices[1]) > 0:
                            y_min, y_max = np.min(mask_indices[0]), np.max(mask_indices[0])
                            x_min, x_max = np.min(mask_indices[1]), np.max(mask_indices[1])
                            
                            # 确保边界框完全在图像内部
                            if (0 <= x_min < x_max < frame.shape[1] and 
                                0 <= y_min < y_max < frame.shape[0]):
                                # 保存无缝克隆前的掩码和图像用于调试
                                debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug")
                                cv2.imwrite(os.path.join(debug_dir, "before_clone_mask.jpg"), mask)
                                cv2.imwrite(os.path.join(debug_dir, "before_clone_result.jpg"), result_img)
                                cv2.imwrite(os.path.join(debug_dir, "before_clone_frame.jpg"), frame)
                                
                                try:
                                    # 确保输入图像是8位无符号整数类型
                                    result_img_8u = result_img.astype(np.uint8)
                                    frame_8u = frame.astype(np.uint8)
                                    mask_8u = mask.astype(np.uint8)
                                    
                                    # 尝试使用无缝克隆
                                    seamless_result = cv2.seamlessClone(
                                        result_img_8u, frame_8u, mask_8u, center_face, cv2.NORMAL_CLONE)
                                        
                                    # 保存无缝克隆后的图像用于调试
                                    cv2.imwrite(os.path.join(debug_dir, "after_clone_result.jpg"), seamless_result)
                                except Exception as e:
                                    print(f"无缝克隆时出错: {e}")
                                    # 如果无缝克隆失败，使用 Poisson 混合
                                    try:
                                        # 使用掩码直接混合
                                        mask_3ch = cv2.merge([mask, mask, mask])
                                        # 归一化掩码
                                        mask_norm = mask_3ch.astype(float) / 255.0
                                        
                                        # 确保数据类型一致
                                        result_img_float = result_img.astype(np.float32)
                                        frame_float = frame.astype(np.float32)
                                        
                                        # 直接混合
                                        seamless_result = (result_img_float * mask_norm + 
                                                          frame_float * (1 - mask_norm)).astype(np.uint8)
                                    except Exception as e2:
                                        print(f"Poisson混合时出错: {e2}")
                                        seamless_result = result_img.astype(np.uint8)
                            else:
                                print("掩码边界框超出图像范围，跳过无缝克隆")
                                seamless_result = result_img.astype(np.uint8)
                        else:
                            print("掩码中没有有效像素，跳过无缝克隆")
                            seamless_result = result_img.astype(np.uint8)
                    else:
                        print("掩码全为零，跳过无缝克隆")
                        seamless_result = result_img.astype(np.uint8)
                except Exception as e:
                    print(f"无缝克隆时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    seamless_result = result_img.astype(np.uint8)
            else:
                print(f"中心点 {center_face} 超出图像范围 {frame.shape[:2]}，跳过无缝克隆")
                seamless_result = result_img.astype(np.uint8)
            
            # 如果启用了颜色校正
            if self.color_correction_var.get():
                try:
                    # 获取面部区域的掩码
                    face_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillConvexPoly(face_mask, hull, 255)
                    
                    # 应用颜色校正
                    seamless_result = self.color_correct(seamless_result, frame, face_mask)
                    # 确保颜色校正后的图像是8位无符号整数类型
                    seamless_result = seamless_result.astype(np.uint8)
                except Exception as e:
                    print(f"颜色校正时出错: {e}")
                    # 如果颜色校正失败，继续使用未校正的结果
                    seamless_result = seamless_result.astype(np.uint8)
            
            # 增强替换效果 - 增加对比度
            try:
                x, y, w, h = cv2.boundingRect(landmarks)
                face_area = seamless_result[y:y+h, x:x+w].copy()
                # 应用对比度增强
                enhanced_face = cv2.addWeighted(face_area, 1.2, face_area, 0, 5)
                seamless_result[y:y+h, x:x+w] = enhanced_face
            except Exception as e:
                print(f"增强对比度时出错: {e}")
            
            # 与原始帧进行对比，确保结果有差异
            seamless_result = seamless_result.astype(np.uint8)
            frame_8u = frame.astype(np.uint8)
            diff = cv2.absdiff(seamless_result, frame_8u)
            diff_mean = np.mean(diff)
            
            if diff_mean < 5:  # 如果差异很小，可能是替换失败
                print(f"警告: 处理前后图像差异很小 ({diff_mean})，可能替换失败")
                # 尝试使用简化方法
                return self.simple_face_swap(frame, rgb_frame, landmarks, target_image, target_landmarks)
            
            # 确保返回值不是原始帧的副本
            if np.array_equal(seamless_result, frame):
                print("警告: 结果与原始帧相同，尝试使用简化替换方法")
                return self.simple_face_swap(frame, rgb_frame, landmarks, target_image, target_landmarks)
            
            # 最终确保返回的是8位无符号整数类型
            return seamless_result.astype(np.uint8)
        except Exception as e:
            print(f"人脸替换过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return self.simple_face_swap(frame, rgb_frame, landmarks, target_image, target_landmarks)  # 出错时使用简化方法
    
    def simple_face_swap(self, frame, rgb_frame, landmarks, target_image, target_landmarks):
        """
        简化的人脸替换方法，使用基本的图像处理技术
        当高级三角剖分方法失败时作为备选方案
        """
        try:
            # 获取源人脸和目标人脸的边界框
            src_face_rect = cv2.boundingRect(landmarks)
            target_face_rect = cv2.boundingRect(target_landmarks)
            
            x, y, w, h = src_face_rect
            x2, y2, w2, h2 = target_face_rect
            
            print(f"简化替换：源人脸位置 ({x},{y},{w},{h}), 目标人脸位置 ({x2},{y2},{w2},{h2})")
            
            # 保存原始帧的副本
            orig_frame = frame.copy()
            
            # 提取源人脸和目标人脸
            src_face = frame[y:y+h, x:x+w]
            target_face = target_image[y2:y2+h2, x2:x2+w2]
            
            # 调整目标人脸大小以匹配源人脸
            target_face_resized = cv2.resize(target_face, (w, h))
            
            # 创建人脸掩码 - 使用椭圆形状
            face_mask = np.zeros((h, w), dtype=np.uint8)
            center = (w // 2, h // 2)
            # 稍微缩小椭圆以避免包含背景
            axes = (int(w * 0.45), int(h * 0.55))
            cv2.ellipse(face_mask, center, axes, 0, 0, 360, 255, -1)
            
            # 应用高斯模糊使边缘平滑
            face_mask = cv2.GaussianBlur(face_mask, (15, 15), 10)
            
            # 将掩码转换为3通道浮点形式
            face_mask_3ch = cv2.merge([face_mask, face_mask, face_mask])
            face_mask_3ch = face_mask_3ch.astype(float) / 255.0
            
            # 确保src_face与target_face_resized的数据类型一致
            src_face = src_face.astype(np.float32)
            target_face_resized = target_face_resized.astype(np.float32)
            
            # 基于掩码融合两个人脸
            blended_face = (target_face_resized * face_mask_3ch + 
                           src_face * (1 - face_mask_3ch))
            blended_face = blended_face.astype(np.uint8)
            
            # 创建输出图像
            result = frame.copy()
            result[y:y+h, x:x+w] = blended_face
            
            # 应用颜色校正
            if self.color_correction_var.get():
                try:
                    # 创建全局掩码
                    global_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    global_mask[y:y+h, x:x+w] = face_mask
                    
                    # 应用颜色校正
                    result = self.enhanced_color_correct(result, orig_frame, global_mask)
                    # 确保颜色校正后的图像是8位无符号整数类型
                    result = result.astype(np.uint8)
                except Exception as e:
                    print(f"颜色校正出错: {e}")
                    # 确保结果是8位无符号整数类型
                    result = result.astype(np.uint8)
            
            # 确保result和orig_frame类型一致后再比较
            result = result.astype(np.uint8)
            orig_frame = orig_frame.astype(np.uint8)
            
            # 使用整体图像的差异来验证替换是否有效
            diff = cv2.absdiff(result, orig_frame)
            mean_diff = np.mean(diff)
            
            print(f"简化替换方法的平均差异: {mean_diff}")
            
            # 如果差异太小，可能替换效果不明显，强化替换效果
            if mean_diff < 5:
                print("简化替换效果不明显，尝试直接复制人脸")
                
                # 直接复制目标人脸到源位置
                # 但仍保持边缘平滑过渡
                global_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.ellipse(global_mask, (x + w//2, y + h//2), 
                           (int(w * 0.45), int(h * 0.55)), 0, 0, 360, 255, -1)
                
                # 应用高斯模糊
                global_mask = cv2.GaussianBlur(global_mask, (31, 31), 11)
                
                # 创建3通道掩码
                mask_3ch = cv2.merge([global_mask, global_mask, global_mask])
                mask_float = mask_3ch.astype(float) / 255.0
                
                # 创建包含目标人脸的图像
                face_area = np.zeros_like(frame, dtype=np.float32)
                face_area[y:y+h, x:x+w] = target_face_resized
                
                # 确保类型一致性
                orig_frame_float = orig_frame.astype(np.float32)
                
                # 在全图范围内进行混合
                result = (face_area * mask_float + 
                         orig_frame_float * (1 - mask_float)).astype(np.uint8)
                
                # 加强效果 - 增加目标人脸的对比度
                face_only = result[y:y+h, x:x+w].copy()
                # 应用对比度增强
                face_contrast = cv2.addWeighted(face_only, 1.2, face_only, 0, 0)
                result[y:y+h, x:x+w] = face_contrast
                
                # 应用颜色校正
                if self.color_correction_var.get():
                    try:
                        result = self.enhanced_color_correct(result, orig_frame, global_mask)
                        # 确保颜色校正后的图像是8位无符号整数类型
                        result = result.astype(np.uint8)
                    except Exception as e:
                        print(f"颜色校正出错: {e}")
                        # 确保结果是8位无符号整数类型
                        result = result.astype(np.uint8)
            
            # 保存调试图像
            debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug")
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "simple_swap_result.jpg"), result)
            cv2.imwrite(os.path.join(debug_dir, "simple_swap_orig.jpg"), orig_frame)
            cv2.imwrite(os.path.join(debug_dir, "simple_swap_mask.jpg"), global_mask if 'global_mask' in locals() else face_mask)
            
            # 最终确保返回的是8位无符号整数类型
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"简化人脸替换方法失败: {e}")
            import traceback
            traceback.print_exc()
            # 失败时返回原始帧，确保是8位无符号整数类型
            return frame.astype(np.uint8)
    
    def enhanced_color_correct(self, target_img, source_img, mask):
        """增强版的颜色校正函数"""
        try:
            # 确保掩码是正确的类型和大小
            if mask.shape[:2] != target_img.shape[:2]:
                print(f"掩码大小 {mask.shape} 与目标图像大小 {target_img.shape[:2]} 不匹配")
                mask = cv2.resize(mask, (target_img.shape[1], target_img.shape[0]))
            
            # 使用掩码获取前景和背景区域
            fg_mask = mask > 10  # 将掩码转换为布尔值
            bg_mask = ~fg_mask
            
            # 检查掩码是否有效
            if not np.any(fg_mask):
                print("颜色校正: 掩码中没有前景区域")
                return target_img.astype(np.uint8)
                
            # 分离RGB通道
            target_channels = cv2.split(target_img)
            source_channels = cv2.split(source_img)
            
            result_channels = []
            
            # 对每个通道单独进行处理
            for i, (target_channel, source_channel) in enumerate(zip(target_channels, source_channels)):
                # 计算前景和背景区域的均值和标准差
                fg_mean, fg_std = cv2.meanStdDev(source_channel, mask=mask)
                target_mean, target_std = cv2.meanStdDev(target_channel, mask=mask)
                
                # 防止除以零或很小的值
                target_std = max(target_std[0][0], 1e-3)
                fg_std = max(fg_std[0][0], 1e-3)
                
                # 对目标图像进行颜色校正
                target_channel = target_channel.astype(np.float32)
                
                # 标准化，然后调整到源图像的分布
                normalized = (target_channel - target_mean[0][0]) / target_std
                corrected = normalized * fg_std + fg_mean[0][0]
                
                # 确保值在有效范围内
                corrected = np.clip(corrected, 0, 255).astype(np.uint8)
                
                # 根据掩码融合校正后的图像和原始图像
                blended = np.zeros_like(target_channel)
                blended[fg_mask] = corrected[fg_mask]
                blended[bg_mask] = target_channel[bg_mask]
                
                result_channels.append(blended)
                
            # 合并通道
            result = cv2.merge(result_channels)
            
            # 确保返回类型是8位无符号整数
            return result.astype(np.uint8)
        except Exception as e:
            print(f"增强颜色校正时出错: {e}")
            import traceback
            traceback.print_exc()
            return target_img.astype(np.uint8)  # 如果出错，返回原始图像

    def find_point_index(self, points, point):
        """在点列表中找到距离给定点最近的点的索引"""
        for i, p in enumerate(points):
            if abs(p[0] - point[0]) < 5 and abs(p[1] - point[1]) < 5:
                return i
        return -1

    def write_frames_in_order(self, out, frames_buffer, next_frame_to_write):
        """按顺序写入已处理的视频帧，返回下一个要写入的帧索引"""
        # 复制一份帧缓冲区的键，并进行排序
        keys = sorted(list(frames_buffer.keys()))
        
        # 确保有需要写入的帧
        if not keys:
            return next_frame_to_write
        
        # 找出连续的帧并写入
        while keys and keys[0] == next_frame_to_write:
            # 写入帧并从缓冲区中移除
            frame = frames_buffer.pop(keys[0])
            
            # 确保帧是有效的
            if frame is not None and frame.size > 0:
                try:
                    out.write(frame)
                except Exception as e:
                    logger.error(f"写入帧 {next_frame_to_write} 时出错: {e}")
            else:
                logger.error(f"帧 {next_frame_to_write} 无效，跳过")
            
            # 更新下一个要写入的帧索引
            next_frame_to_write += 1
            
            # 从排序后的键列表中移除已处理的键
            keys.pop(0)
        
        # 更新进度条
        if hasattr(self, 'progress_var'):
            # 安全获取total_frames属性
            total_frames = getattr(self, 'total_frames', 0)
            if total_frames > 0:
                progress_value = min(100, int((next_frame_to_write / total_frames) * 100))
                self.progress_var.set(progress_value)
                # 更新进度标签
                progress_text = f"{progress_value}%"
                self.root.after(0, lambda t=progress_text: self.progress_label.config(text=t))
                
                # 更新状态文本
                status_text = f"处理中... {progress_value}%"
                self.root.after(0, lambda t=status_text: self.status_var.set(t))
                
        return next_frame_to_write

    def shape_to_np(self, shape):
        """将dlib的shape转换为numpy数组"""
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def color_correct(self, target_img, source_img, mask):
        """对目标图像进行颜色校正，使其与源图像的颜色匹配"""
        try:
            # 确保掩码是正确的类型和大小
            if mask.shape[:2] != target_img.shape[:2]:
                print(f"掩码大小 {mask.shape} 与目标图像大小 {target_img.shape[:2]} 不匹配")
                # 调整掩码大小以匹配目标图像
                mask = cv2.resize(mask, (target_img.shape[1], target_img.shape[0]))
            
            # 确保掩码是 CV_8U 类型
            mask = mask.astype(np.uint8)
            
            # 转换为LAB颜色空间
            target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB)
            source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)
            
            # 分离通道
            target_l, target_a, target_b = cv2.split(target_lab)
            source_l, source_a, source_b = cv2.split(source_lab)
            
            # 计算源图像的统计数据
            source_l_mean, source_l_std = cv2.meanStdDev(source_l, mask=mask)
            source_a_mean, source_a_std = cv2.meanStdDev(source_a, mask=mask)
            source_b_mean, source_b_std = cv2.meanStdDev(source_b, mask=mask)
            
            # 计算目标图像的统计数据
            target_l_mean, target_l_std = cv2.meanStdDev(target_l, mask=mask)
            target_a_mean, target_a_std = cv2.meanStdDev(target_a, mask=mask)
            target_b_mean, target_b_std = cv2.meanStdDev(target_b, mask=mask)
            
            # 防止除以零
            target_l_std = np.maximum(target_l_std, 1e-5)
            target_a_std = np.maximum(target_a_std, 1e-5)
            target_b_std = np.maximum(target_b_std, 1e-5)
            
            # 调整目标图像的亮度和颜色 - 确保使用 float32 而不是 float64
            target_l = target_l.astype(np.float32)
            target_a = target_a.astype(np.float32)
            target_b = target_b.astype(np.float32)
            
            # 计算调整后的值
            l_adjusted = ((target_l - target_l_mean[0][0]) * (source_l_std[0][0] / target_l_std[0][0])) + source_l_mean[0][0]
            a_adjusted = ((target_a - target_a_mean[0][0]) * (source_a_std[0][0] / target_a_std[0][0])) + source_a_mean[0][0]
            b_adjusted = ((target_b - target_b_mean[0][0]) * (source_b_std[0][0] / target_b_std[0][0])) + source_b_mean[0][0]
            
            # 确保值在有效范围内
            l_adjusted = np.clip(l_adjusted, 0, 255).astype(np.uint8)
            a_adjusted = np.clip(a_adjusted, 0, 255).astype(np.uint8)
            b_adjusted = np.clip(b_adjusted, 0, 255).astype(np.uint8)
            
            # 合并通道
            target_lab = cv2.merge([l_adjusted, a_adjusted, b_adjusted])
            
            # 转换回BGR颜色空间
            target_corrected = cv2.cvtColor(target_lab, cv2.COLOR_LAB2BGR)
            
            # 使用掩码应用颜色校正
            # 创建三通道掩码
            mask_3channel = np.stack([mask, mask, mask], axis=2)
            
            # 确保掩码和图像大小匹配
            if mask_3channel.shape != target_corrected.shape:
                print(f"三通道掩码大小 {mask_3channel.shape} 与目标图像大小 {target_corrected.shape} 不匹配")
                return target_img.astype(np.uint8)
            
            # 使用掩码应用颜色校正
            result = np.zeros_like(target_img)
            cv2.bitwise_and(target_corrected, mask_3channel, result)
            
            # 创建反向掩码
            inverse_mask = cv2.bitwise_not(mask)
            inverse_mask_3channel = np.stack([inverse_mask, inverse_mask, inverse_mask], axis=2)
            
            # 应用反向掩码到原始图像
            background = np.zeros_like(target_img)
            cv2.bitwise_and(target_img, inverse_mask_3channel, background)
            
            # 合并结果
            final_result = cv2.add(result, background)
            
            # 确保返回类型是8位无符号整数
            return final_result.astype(np.uint8)
        except Exception as e:
            print(f"颜色校正时出错: {e}")
            import traceback
            traceback.print_exc()
            return target_img.astype(np.uint8)  # 如果出错，返回原始图像

    def preview_frame(self):
        """预览当前帧的人脸替换效果"""
        if not self.video_path or not self.face_images:
            messagebox.showerror("错误", "请先选择视频和人脸图片")
            return
        
        # 创建预览窗口
        preview_window = tk.Toplevel(self.root)
        preview_window.title("预览")
        preview_window.geometry("800x600")
        
        # 创建进度标签
        progress_label = ttk.Label(preview_window, text="正在处理...", style="TLabel")
        progress_label.pack(pady=10)
        
                # 在后台线程中处理图像
        def process_preview():
            try:
                # 获取视频的中间帧
                cap = cv2.VideoCapture(self.video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            middle_frame = total_frames // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                ret, frame = cap.read()
                cap.release()
            
            if not ret:
                messagebox.showerror("错误", "无法读取视频帧")
                    preview_window.destroy()
                return
            
                # 显示原始图像
                original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                original_img = Image.fromarray(original_frame)
                original_img = self.resize_image_aspect_ratio(original_img, 380, 300)
                original_photo = ImageTk.PhotoImage(original_img)
                
                # 创建原始图像标签
                original_label = ttk.Label(preview_window, image=original_photo)
                original_label.image = original_photo
                original_label.pack(side=tk.LEFT, padx=10, pady=10)
                
                # 处理图像
                processed_frame = self.process_frame(frame)
                
                if processed_frame is not None:
                    # 显示处理后的图像
                    processed_img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                    processed_img = self.resize_image_aspect_ratio(processed_img, 380, 300)
                    processed_photo = ImageTk.PhotoImage(processed_img)
                    
                    # 创建处理后图像标签
                    processed_label = ttk.Label(preview_window, image=processed_photo)
                    processed_label.image = processed_photo
                    processed_label.pack(side=tk.RIGHT, padx=10, pady=10)
                    
                    # 添加保存按钮
                    def save_preview_image():
                        file_path = filedialog.asksaveasfilename(
                            defaultextension=".png",
                            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
                        )
                        if file_path:
                            cv2.imwrite(file_path, processed_frame)
                            messagebox.showinfo("成功", "预览图片已保存")
                    
                    save_btn = ttk.Button(preview_window, text="保存预览图片", 
                                        command=save_preview_image,
                                        style="TButton")
                    save_btn.pack(pady=10)
                    
                    progress_label.config(text="处理完成")
                    else:
                    progress_label.config(text="处理失败 - 未能检测到人脸或应用替换")
        except Exception as e:
                logger.error(f"预览处理错误: {str(e)}")
                progress_label.config(text=f"处理错误: {str(e)}")
    
        # 开始处理线程
        threading.Thread(target=process_preview, daemon=True).start()
    
    def resize_image_aspect_ratio(self, image, max_width, max_height):
        """保持宽高比调整图像大小"""
        width, height = image.size
        # 确定缩放比例，保持宽高比
        scale = min(max_width/width, max_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return image.resize((new_width, new_height), Image.LANCZOS)

    def process_frame_traditional(self, frame, target_image_rgb, target_landmarks, detector_choice, use_multi_scale):
        """使用传统方法处理单个视频帧"""
        try:
            # 转换帧到RGB格式以供处理
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 检测人脸
            landmarks = None
            
            if detector_choice == "dlib":
                # 使用dlib人脸检测器
                upsample = 1 if use_multi_scale else 0
                faces = self.detector(rgb_frame, upsample)
                
                if len(faces) == 0:
                    # 如果未检测到人脸，返回原始帧
                    return frame
                
                # 从第一个检测到的人脸获取特征点
                try:
                    shape = self.predictor(rgb_frame, faces[0])
                    landmarks = self.shape_to_np(shape)
                except Exception as lm_e:
                    logger.error(f"dlib提取特征点失败: {lm_e}")
                    return frame
            else:
                # 使用OpenCV人脸检测器
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) == 0:
                    # 如果未检测到人脸，返回原始帧
                    return frame
                
                # 从第一个检测到的人脸获取特征点
                try:
                    (x, y, w, h) = faces[0]
                    face_gray = gray[y:y+h, x:x+w]
                    face_roi = rgb_frame[y:y+h, x:x+w]
                    
                    # 首先尝试使用dlib特征点检测
                    rect = dlib.rectangle(x, y, x+w, y+h)
                    shape = self.predictor(rgb_frame, rect)
                    landmarks = self.shape_to_np(shape)
                except Exception as cv_lm_e:
                    logger.error(f"OpenCV+dlib提取特征点失败: {cv_lm_e}")
                    return frame
            
            # 如果未找到有效的特征点，返回原始帧
            if landmarks is None or len(landmarks) == 0:
                logger.warning("未能提取有效的面部特征点")
                return frame
            
            # 选择换脸方法 - 高级或简单
            try:
                if self.swap_method_var.get() == "advanced":
                    result = self.advanced_face_swap(frame, rgb_frame, landmarks, target_image, target_landmarks)
                else:
                    result = self.simple_face_swap(frame, rgb_frame, landmarks, target_image, target_landmarks)
                
                # 应用颜色校正
                if self.color_correction_var.get():
                    # 创建基于特征点的掩码
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    hull = cv2.convexHull(landmarks)
                    cv2.fillConvexPoly(mask, hull, 255)
                    
                    # 确保掩码有效
                    if np.sum(mask) > 0:
                        result = self.color_correct(result, frame, mask)
                
                return result
            except Exception as swap_e:
                logger.error(f"执行人脸替换时出错: {swap_e}")
                import traceback
                traceback.print_exc()
                return frame
                
        except Exception as e:
            logger.error(f"传统方法处理帧时出错: {e}")
            import traceback
            traceback.print_exc()
            return frame

def main():
    try:
        # 确保环境变量已设置
        if 'TCL_IGNORE_VERSION_CHECK' not in os.environ:
            os.environ['TCL_IGNORE_VERSION_CHECK'] = '1'
        if 'TCL_LIBRARY' not in os.environ:
            os.environ['TCL_LIBRARY'] = os.path.join(python_path, 'Library', 'lib', 'tcl8.6')
        if 'TK_LIBRARY' not in os.environ:
            os.environ['TK_LIBRARY'] = os.path.join(python_path, 'Library', 'lib', 'tk8.6')
            
        # 尝试修改 init.tcl 文件中的版本要求
        try:
            init_tcl_path = os.path.join(python_path, 'Library', 'lib', 'tcl8.6', 'init.tcl')
            if os.path.exists(init_tcl_path):
                with open(init_tcl_path, 'r') as f:
                    content = f.read()
                
                # 创建备份
                if not os.path.exists(init_tcl_path + '.bak'):
                    with open(init_tcl_path + '.bak', 'w') as f:
                        f.write(content)
                
                # 修改版本要求
                modified_content = content.replace('package require -exact Tcl 8.6.13', 
                                                 'package require -exact Tcl 8.6.12')
                
                with open(init_tcl_path, 'w') as f:
                    f.write(modified_content)
                
                print("已修改 init.tcl 文件以适配 Tcl 8.6.12 版本")
            
            # 修改 tk.tcl 文件中的版本要求
            tk_tcl_path = os.path.join(python_path, 'Library', 'lib', 'tk8.6', 'tk.tcl')
            if os.path.exists(tk_tcl_path):
                with open(tk_tcl_path, 'r') as f:
                    content = f.read()
                
                # 创建备份
                if not os.path.exists(tk_tcl_path + '.bak'):
                    with open(tk_tcl_path + '.bak', 'w') as f:
                        f.write(content)
                
                # 修改版本要求
                modified_content = content.replace('package require -exact Tk  8.6.13', 
                                                 'package require -exact Tk  8.6.12')
                
                with open(tk_tcl_path, 'w') as f:
                    f.write(modified_content)
                
                print("已修改 tk.tcl 文件以适配 Tk 8.6.12 版本")
            
        except Exception as e:
            print(f"修改 TCL/TK 文件时出错: {e}")
            print("将尝试继续启动应用...")
        
        # 创建 Tk 实例前强制设置环境变量
        import tkinter as tk
        tk._default_root = None
        
        # 创建应用程序
        root = tk.Tk()
        
        try:
            app = FaceSwapApp(root)
            root.mainloop()
        except Exception as e:
            print(f"初始化应用时出错: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("错误", f"初始化应用时出错: {e}")
    except Exception as e:
        print(f"启动应用程序时出错: {e}")
        print("详细错误信息:")
        import traceback
        traceback.print_exc()
        input("按回车键继续...")

if __name__ == "__main__":
    # 检查环境
    try:
        # 首先导入所需的库，检查它们是否可用
        import_modules = ["dlib", "cv2", "PIL", "moviepy"]
        missing_modules = []
        
        for module in import_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            print(f"错误: 缺少必要的库: {', '.join(missing_modules)}")
            print("请确保已安装所有必要的库")
            sys.exit(1)
        else:
            print("环境检查通过，所有必要的库都已安装")
    except Exception as e:
        print(f"环境检查时出错: {e}")
        sys.exit(1)
        
    # 启动应用
    main()