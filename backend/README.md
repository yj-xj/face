# 人脸替换应用 - 后端服务器

## 快速开始

### 1. 安装 MySQL

确保已安装 MySQL 5.7+ 或 MariaDB 10.3+

### 2. 创建数据库

```bash
mysql -u root -p < database.sql
```

### 3. 安装 Python 依赖

```bash
cd backend
pip install -r requirements.txt
```

### 4. 配置数据库连接

编辑 `app.py` 中的数据库配置：

```python
config = {
    'DATABASE_URL': 'mysql+pymysql://用户名:密码@localhost/face_swap_db'
}
```

### 5. 启动服务器

```bash
python app.py
```

服务器将在 `http://localhost:5000` 启动

### 6. 测试 API

```bash
# 健康检查
curl http://localhost:5000/api/health

# 上传图片
curl -X POST -F "file=@test.jpg" -F "user_id=1" http://localhost:5000/api/upload/image

# 上传视频
curl -X POST -F "file=@test.mp4" -F "user_id=1" http://localhost:5000/api/upload/video

# 获取图片列表
curl http://localhost:5000/api/images

# 获取视频列表
curl http://localhost:5000/api/videos
```

## API 文档

### 健康检查
- **GET** `/api/health`
- 返回服务器状态

### 上传图片
- **POST** `/api/upload/image`
- 参数: `file` (文件), `user_id` (可选)
- 返回: 图片信息

### 上传视频
- **POST** `/api/upload/video`
- 参数: `file` (文件), `user_id` (可选)
- 返回: 视频信息

### 获取图片列表
- **GET** `/api/images`
- 参数: `user_id`, `page`, `per_page`
- 返回: 图片列表

### 获取视频列表
- **GET** `/api/videos`
- 参数: `user_id`, `page`, `per_page`
- 返回: 视频列表

### 获取图片文件
- **GET** `/api/images/<id>`
- 返回: 图片文件

### 获取视频文件
- **GET** `/api/videos/<id>`
- 返回: 视频文件

### 获取缩略图
- **GET** `/api/thumbnails/<id>`
- 返回: 缩略图文件

### 删除图片
- **DELETE** `/api/images/<id>`
- 返回: 删除结果

### 删除视频
- **DELETE** `/api/videos/<id>`
- 返回: 删除结果

## 文件结构

```
backend/
├── app.py              # Flask 应用
├── database.sql        # 数据库表结构
├── requirements.txt    # Python 依赖
└── README.md          # 本文件

uploads/               # 上传文件目录
├── videos/            # 视频文件
├── images/            # 图片文件
├── output/            # 输出视频
└── thumbnails/        # 缩略图
```

## 数据库表

- `users` - 用户表
- `face_images` - 人脸图片表
- `input_videos` - 输入视频表
- `output_videos` - 输出视频表
- `processing_tasks` - 处理任务表
- `system_config` - 系统配置表

## 性能优化

- 支持大文件上传（最大500MB）
- 自动生成缩略图
- 视频信息提取
- 分页查询
- 索引优化

## 故障排除

### 数据库连接失败
- 检查 MySQL 服务是否运行
- 检查用户名密码是否正确
- 检查数据库是否已创建

### 上传失败
- 检查文件大小是否超过限制
- 检查文件格式是否支持
- 检查 uploads 目录权限

## 注意事项

1. 本开发版使用默认用户ID=1
2. 生产环境需要添加用户认证
3. 建议使用 Nginx 作为反向代理
4. 定期清理 uploads 目录
