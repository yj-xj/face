# Django 人脸替换后端服务

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置数据库

编辑 `face_swap/settings.py` 中的数据库配置：

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'face_swap_db',
        'USER': 'root',
        'PASSWORD': 'your_password',
        'HOST': 'localhost',
        'PORT': '3306',
    }
}
```

### 3. 运行迁移

```bash
python manage.py makemigrations
python manage.py migrate
```

### 4. 创建超级用户

```bash
python manage.py createsuperuser
```

### 5. 启动服务器

```bash
python manage.py runserver 0.0.0.0:8000
```

### 6. 访问 API

- API 文档: http://localhost:8000/api/docs/
- Admin 后台: http://localhost:8000/admin/
- API 端点: http://localhost:8000/api/v1/

## API 端点

### 认证
- `POST /api/v1/auth/register/` - 用户注册
- `POST /api/v1/auth/login/` - 用户登录
- `POST /api/v1/auth/logout/` - 用户登出

### 人脸图片
- `GET /api/v1/images/` - 获取图片列表
- `POST /api/v1/images/` - 上传图片
- `GET /api/v1/images/{id}/` - 获取图片详情
- `GET /api/v1/images/{id}/file/` - 下载图片文件
- `DELETE /api/v1/images/{id}/` - 删除图片

### 视频文件
- `GET /api/v1/videos/` - 获取视频列表
- `POST /api/v1/videos/` - 上传视频
- `GET /api/v1/videos/{id}/` - 获取视频详情
- `GET /api/v1/videos/{id}/file/` - 下载视频文件
- `DELETE /api/v1/videos/{id}/` - 删除视频

### 视频处理
- `POST /api/v1/process/` - 提交处理任务
- `GET /api/v1/process/{id}/` - 获取任务状态
- `GET /api/v1/process/` - 获取任务列表

## 部署

### 生产环境配置

1. 设置 `DEBUG = False`
2. 配置 `ALLOWED_HOSTS`
3. 使用 Gunicorn + Nginx
4. 配置静态文件服务

### 环境变量

```bash
export DJANGO_SECRET_KEY='your-secret-key'
export DATABASE_URL='mysql://user:password@localhost/face_swap_db'
export ALLOWED_HOSTS='localhost,yourdomain.com'
```

## 开发

### 运行测试

```bash
python manage.py test
```

### 生成文档

```bash
pip install drf-yasg
# 访问 http://localhost:8000/api/docs/
```
