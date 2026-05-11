
"""
前端配置文件
"""
from decouple import config

# API 配置
API_BASE_URL = config('API_BASE_URL', default='http://localhost:8008/api')
API_TIMEOUT = config('API_TIMEOUT', default=30, cast=int)
UPLOAD_TIMEOUT = config('UPLOAD_TIMEOUT', default=600, cast=int)  # 10分钟
DOWNLOAD_TIMEOUT = config('DOWNLOAD_TIMEOUT', default=300, cast=int)  # 5分钟

# 文件配置
MAX_UPLOAD_SIZE = config('MAX_UPLOAD_SIZE', default=524288000, cast=int)  # 500MB
ALLOWED_VIDEO_EXTENSIONS = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv']
ALLOWED_IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp']

# UI 配置
WINDOW_WIDTH = config('WINDOW_WIDTH', default=1600, cast=int)
WINDOW_HEIGHT = config('WINDOW_HEIGHT', default=900, cast=int)
THEME = config('THEME', default='dark')

# 视频配置
DEFAULT_VIDEO_FPS = config('DEFAULT_VIDEO_FPS', default=30, cast=int)
CAMERA_WIDTH = config('CAMERA_WIDTH', default=640, cast=int)
CAMERA_HEIGHT = config('CAMERA_HEIGHT', default=480, cast=int)

# 处理配置
DEFAULT_PROCESSING_METHOD = config('DEFAULT_PROCESSING_METHOD', default='inswapper')
ENABLE_COLOR_CORRECTION = config('ENABLE_COLOR_CORRECTION', default=True, cast=bool)
ENABLE_MULTI_SCALE = config('ENABLE_MULTI_SCALE', default=True, cast=bool)
