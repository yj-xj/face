# 人脸替换应用 (Face Swap Application)

基于 PyQt5 和 Django 的人脸替换应用程序，支持视频和实时摄像头模式。

## 项目结构

```
face/
├── backend-django/          # Django 后端服务
│   ├── api/                # REST API 接口
│   ├── core/               # 数据模型
│   ├── face_swap/          # Django 配置
│   ├── manage.py           # Django 管理脚本
│   └── requirements.txt    # Python 依赖
│
├── frontend/               # PyQt5 前端应用
│   ├── config.py           # 前端配置
│   ├── database_manager.py # 数据库管理器
│   ├── face_swap.py        # 核心处理模块
│   ├── face_swap_ui_enhanced.py  # 主界面
│   ├── main.py             # 应用入口
│   └── requirements.txt    # Python 依赖
│
├── models/                 # AI 模型文件
│   ├── inswapper_128.onnx  # 人脸替换模型
│   └── buffalo_l/          # 人脸检测模型
│
├── data/                   # 输入数据目录
│   ├── input_faces/        # 人脸图片
│   └── input_videos/       # 输入视频
│
├── output_videos/          # 输出视频目录
├── logs/                   # 日志文件
├── start_backend.bat       # 启动后端脚本
├── start_frontend.bat      # 启动前端脚本
└── README.md              # 项目说明
```

## 功能特性

### 核心功能
- ✅ **视频模式** - 上传视频进行人脸替换
- ✅ **摄像头模式** - 实时人脸替换（优化至 <50ms 延迟）
- ✅ **多种处理方法** - 传统方法 + InsightFace AI
- ✅ **上传功能** - 将视频和图片上传到数据库
- ✅ **数据库管理** - 保存和管理所有处理的文件

### 性能优化
- 🚀 摄像头延迟降低至 <50ms
- 🚀 处理帧率提升至 40-50 FPS
- 🚀 支持多尺度人脸检测
- 🚀 颜色校正和平滑处理

## 快速开始

### 环境要求

- Python 3.9+
- CUDA (可选，用于 GPU 加速)

### 安装依赖

**后端：**
```bash
cd backend-django
pip install -r requirements.txt
python manage.py migrate
```

**前端：**
```bash
cd frontend
pip install -r requirements.txt
```

### 启动应用

#### 方法 1：使用批处理文件（Windows）

1. 双击 `start_backend.bat` 启动后端服务
2. 双击 `start_frontend.bat` 启动前端界面

#### 方法 2：使用命令行

**终端 1 - 启动后端：**
```bash
cd backend-django
python manage.py runserver 0.0.0.0:8000
```

**终端 2 - 启动前端：**
```bash
cd frontend
python main.py
```

### 访问地址

- **前端界面**：自动弹出 PyQt 窗口
- **API 文档**：http://localhost:8000/api/docs/
- **管理后台**：http://localhost:8000/admin/
- **REST API**：http://localhost:8000/api/v1/

## API 端点

### 人脸图片
- `GET /api/v1/images/` - 获取所有图片
- `POST /api/v1/images/` - 上传图片
- `GET /api/v1/images/{id}/` - 获取图片详情
- `DELETE /api/v1/images/{id}/` - 删除图片

### 视频管理
- `GET /api/v1/videos/` - 获取所有视频
- `POST /api/v1/videos/` - 上传视频
- `GET /api/v1/videos/{id}/` - 获取视频详情
- `DELETE /api/v1/videos/{id}/` - 删除视频

### 处理任务
- `GET /api/v1/tasks/` - 获取所有任务
- `POST /api/v1/tasks/` - 创建处理任务
- `GET /api/v1/tasks/{id}/` - 获取任务状态
- `POST /api/v1/tasks/{id}/cancel/` - 取消任务

## 使用说明

### 视频模式

1. 点击"选择视频"按钮选择视频文件
2. 点击"选择人脸图片"上传目标人脸
3. 选择处理方法（Inswapper 或 Traditional）
4. 点击"开始处理"开始人脸替换
5. 处理完成后可播放和下载结果

### 摄像头模式

1. 点击"切换到摄像头"按钮
2. 选择要替换的人脸图片
3. 点击"开启摄像头"开始实时替换
4. 可随时截图保存当前帧

### 上传功能

1. 点击"[上传] 添加视频"或上传图片按钮
2. 选择文件后自动上传到数据库
3. 上传的文件可在界面中查看和管理

## 技术栈

### 后端
- Django 4.2.7 - Web 框架
- Django REST Framework 3.14.0 - API 框架
- SQLite/MySQL - 数据库
- OpenCV - 图像处理
- InsightFace - 人脸识别

### 前端
- PyQt5 - GUI 框架
- OpenCV - 视频处理
- NumPy - 数值计算
- Pillow - 图像处理

## 配置说明

### 前端配置 (`frontend/config.py`)

```python
API_BASE_URL = "http://localhost:8000/api/v1"  # 后端 API 地址
MAX_UPLOAD_SIZE = 524288000  # 最大上传大小 (500MB)
ALLOWED_VIDEO_EXTENSIONS = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv']
ALLOWED_IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp']
```

### 后端配置 (`backend-django/face_swap/settings.py`)

```python
DEBUG = True  # 开发模式
ALLOWED_HOSTS = ['localhost', '127.0.0.1']
DATABASES = {...}  # 数据库配置（默认 SQLite）
```

## 故障排除

### 问题：无法连接后端
- 检查后端是否正常启动
- 确认端口 8000 未被占用
- 检查防火墙设置

### 问题：摄像头无法打开
- 确认摄像头未被其他应用占用
- 检查摄像头驱动是否正常
- 尝试重启应用

### 问题：处理速度慢
- 确认 GPU 驱动已安装（如使用 CUDA）
- 降低视频分辨率
- 关闭颜色校正功能

## 开发计划

- [ ] 添加用户认证系统
- [ ] 支持批量视频处理
- [ ] 添加更多人脸替换效果
- [ ] 优化大文件上传性能
- [ ] 添加实时预览功能

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v3.0.0 (2025-12-31)
- ✨ 前后端分离架构
- ✨ 添加数据库管理功能
- ✨ 优化摄像头性能（<50ms 延迟）
- ✨ 全新白色科技感界面
- 🐛 修复 OpenCV GaussianBlur 错误
- 📝 整理项目文件结构

### v2.0.0 (2025-12-30)
- 🎉 基础人脸替换功能
- ✨ 视频和摄像头模式
