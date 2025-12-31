# 人脸替换应用 (Face Swap Application)

基于 InsightFace 和传统算法的人脸替换应用，支持���频处理和实时摄像头换脸。

## 功能特点

- **双模式支持**
  - 📹 **视频模式**: 批量处理视频文件，支持人脸替换
  - 📷 **摄像头模式**: 实时摄像头人脸检测和替换

- **多种换脸算法**
  - ✨ **Inswapper**: 基于 InsightFace 的深度学习换脸模型，效果自然
  - 🔧 **传统方法**: 基于 Delaunay 三角剖分的传统算法

- **高级功能**
  - 颜色校正
  - 多尺度人脸检测
  - 多种人脸检测器（Dlib、OpenCV）
  - 平滑度调节
  - 实时预览和拍照保存

## 环境要求

- Python 3.9+
- Windows 操作系统
- 摄像头（用于实时模式）

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/yj-xj/face.git
cd face
```

### 2. 创建虚拟环境（推荐）

```bash
conda create -n face_swap python=3.9
conda activate face_swap
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- PyQt5
- OpenCV
- InsightFace
- dlib
- onnxruntime
- moviepy
- numpy
- Pillow

## 模型文件

### 必需的模型文件

应用需要以下模型文件才能正常工作：

1. **Inswapper 模型** (529MB)
   - 下载地址: https://huggingface.co/deepinsight/inswapper/blob/main/inswapper_128.onnx
   - 安装位置: `models/inswapper_128.onnx`

2. **Dlib 特征点模型** (99MB)
   - 下载地址: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   - 安装位置: `models/shape_predictor_68_face_landmarks.dat`

3. **Buffalo_l 模型** (人脸检测，约 220MB)
   - 应用会自动下载到 `models/buffalo_l/` 目录

### 快速下载脚本

```python
import os
from huggingface_hub import hf_hub_download

# 下载 inswapper 模型
os.makedirs('models', exist_ok=True)
hf_hub_download(
    repo_id='deepinsight/inswapper',
    filename='inswapper_128.onnx',
    local_dir='models',
    local_dir_use_symlinks=False
)
print("模型下载完成!")
```

## 使用方法

### 启动应用

```bash
cd src
python face_swap_ui_enhanced.py
```

### 视频模式

1. 选择要替换的人脸图片（支持多张）
2. 选择要处理的视频文件
3. 配置高级选项（可选）
   - 面部平滑度
   - 颜色校正
   - 多尺度检测
   - 检测器类型
   - 换脸算法
4. 设置输出路径
5. 点击"开始处理"

### 摄像头模式

1. 切换到"摄像头模式"
2. 选择要替换的人脸图片
3. 点击"开启摄像头"
4. 启用"实时人脸替换"开关
5. 可随时拍照保存

## 性能优化

### 摄像头模式优化

- **跳帧处理**: 自动跳帧以提高流畅度
- **分辨率降低**: 对于高分辨率摄像头，自动降低处理分辨率
- **快速模式**: 使用 FastTransformation 替代 SmoothTransformation

### 视频处理优化

- **进度节流**: 进度更新频率降低到 0.5 秒/次
- **并行处理**: 支持多线程并行处理帧
- **内存优化**: 共享缩略图，减少内存占用

## 故障排除

### 问题 1: 摄像头延迟

**解决方案**:
- 应用已自动优化：
  - 跳帧处理（每 N 帧处理一次）
  - 自动降低高分辨率视频的处理分辨率
  - 使用快速变换算法

### 问题 2: 传统方法崩溃

**解决方案**:
- 已修复 `StringVar` 兼容性问题
- 添加了完善的错误处理
- 确保 Dlib 模型文件存在

### 问题 3: Inswapper 不可用

**解决方案**:
1. 检查 `models/inswapper_128.onnx` 是否存在
2. 检查文件大小是否为 529MB 左右
3. 如果文件损坏，重新下载

### 问题 4: 人脸检测失败

**解决方案**:
- 确保图片中人脸清晰可见
- 尝试启用"多尺度检测"
- 切换检测器（Dlib 或 OpenCV）
- 改善光线条件

## 项目结构

```
face/
├── src/                          # 源代码目录
│   ├── face_swap.py             # 核心换脸逻辑
│   ├── face_swap_ui_enhanced.py # PyQt5 增强界面
│   └── ignore_ssl_warnings.py   # SSL 忽略模块
├── models/                       # 模型文件目录
│   ├── inswapper_128.onnx       # Inswapper 模型
│   ├── shape_predictor_68_*.dat # Dlib 模型
│   └── buffalo_l/               # 人脸检测模型
├── data/                         # 数据目录
│   ├── input_faces/             # 输入人脸图片
│   └── input_videos/            # 输入视频
├── output_videos/                # 输出视频目录
├── logs/                         # 日志目录
├── README.md                     # 本文件
└── requirements.txt              # 依赖列表
```

## 技术架构

### 核心技术栈

- **GUI**: PyQt5 (增强界面) / Tkinter (原始界面)
- **计算机视觉**: OpenCV, Dlib
- **深度学习**: InsightFace, ONNX Runtime
- **视频处理**: MoviePy

### 算法说明

#### Inswapper 算法
- 基于 InsightFace 的深度学习模型
- 使用 buffalo_l 进行人脸检测和特征提取
- 使用 inswapper_128.onnx 进行人脸替换
- 效果自然，但需要 GPU 才能达到实时性能

#### 传统三角剖分算法
- 使用 Dlib 检测 68 个面部特征点
- Delaunay 三角剖分进行面部区域划分
- 三角形变换实现人脸替换
- 适合 CPU 处理，但效果略差

## 性能基准

### 摄像头模式
- **帧率**: 20-30 FPS (优化后)
- **延迟**: < 100ms (优化后)
- **CPU 占用**: 30-50%

### 视频处理
- **处理速度**: 10-15 fps (Inswapper)
- **处理速度**: 20-30 fps (传统方法)
- **内存占用**: 400-800MB

## 更新日志

### v2.3 (2025-12-31)
- ✅ 修复摄像头实时换脸延迟问题
  - 添加跳帧处理
  - 自动降低高分辨率处理
  - 移除阻塞式 sleep

- ✅ 修复传统三角剖分方法崩溃
  - 修复 StringVar 兼容性问题
  - 添加完善的错误处理

- ✅ 优化性能
  - 使用 FastTransformation
  - 减少进度更新频率
  - 优化内存使用

- ✅ UI 改进
  - 独立的视频和摄像头控制栏
  - 科技感美化动效
  - 状态指示器

### v2.2 (2025-12-31)
- 性能优化报告
- 视频播放优化 (300% 提升)
- CPU 占用降低 70%

### v2.1 (2025-12-31)
- UI 界面完全分离
- 摄像头模式使用 Inswapper

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 联系方式

- GitHub: @yj-xj
- 项目地址: https://github.com/yj-xj/face

## 致谢

- [InsightFace](https://github.com/deepinsight/insightface) - 优秀的人脸识别库
- [Dlib](http://dlib.net/) - C++ 工具包，包含人脸检测
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - Python GUI 框架
