#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
人脸替换应用后端服务
Flask API + MySQL
"""
import os
import sys
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import cv2
import numpy as np

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置
config = {
    'UPLOAD_FOLDER': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads'),
    'MAX_CONTENT_LENGTH': 500 * 1024 * 1024,  # 500MB
    'ALLOWED_VIDEO_EXTENSIONS': {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'},
    'ALLOWED_IMAGE_EXTENSIONS': {'jpg', 'jpeg', 'png', 'bmp'},
    'DATABASE_URL': 'mysql+pymysql://root:password@localhost/face_swap_db'
}

app.config['UPLOAD_FOLDER'] = config['UPLOAD_FOLDER']
app.config['MAX_CONTENT_LENGTH'] = config['MAX_CONTENT_LENGTH']
app.config['SQLALCHEMY_DATABASE_URI'] = config['DATABASE_URL']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'videos'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'output'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'thumbnails'), exist_ok=True)

# 数据库初始化
db = SQLAlchemy(app)

# ============================================================================
# 数据库模型
# ============================================================================

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.TIMESTAMP, default=datetime.utcnow)
    updated_at = db.Column(db.TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)

class FaceImage(db.Model):
    __tablename__ = 'face_images'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.BigInteger, nullable=False)
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    thumbnail_path = db.Column(db.String(500))
    face_count = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.TIMESTAMP, default=datetime.utcnow)
    updated_at = db.Column(db.TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)

class InputVideo(db.Model):
    __tablename__ = 'input_videos'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.BigInteger, nullable=False)
    duration = db.Column(db.Float)
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    fps = db.Column(db.Float)
    codec = db.Column(db.String(50))
    thumbnail_path = db.Column(db.String(500))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.TIMESTAMP, default=datetime.utcnow)
    updated_at = db.Column(db.TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)

class OutputVideo(db.Model):
    __tablename__ = 'output_videos'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    input_video_id = db.Column(db.Integer, db.ForeignKey('input_videos.id', ondelete='SET NULL'))
    face_image_id = db.Column(db.Integer, db.ForeignKey('face_images.id', ondelete='CASCADE'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.BigInteger, nullable=False)
    duration = db.Column(db.Float)
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    fps = db.Column(db.Float)
    processing_method = db.Column(db.Enum('traditional', 'inswapper'), default='inswapper')
    processing_time = db.Column(db.Float)
    thumbnail_path = db.Column(db.String(500))
    status = db.Column(db.Enum('processing', 'completed', 'failed'), default='processing')
    error_message = db.Column(db.Text)
    created_at = db.Column(db.TIMESTAMP, default=datetime.utcnow)
    completed_at = db.Column(db.TIMESTAMP)

# ============================================================================
# 工具函数
# ============================================================================

def allowed_file(filename, allowed_extensions):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def generate_unique_filename(original_filename):
    """生成唯一文件名"""
    ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    return unique_name

def create_thumbnail(video_path, thumbnail_path, time_seconds=1):
    """创建视频缩略图"""
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (320, 240))
            cv2.imwrite(thumbnail_path, frame)
            cap.release()
            return True
        cap.release()
        return False
    except Exception as e:
        print(f"创建缩略图失败: {e}")
        return False

def get_video_info(video_path):
    """获取视频信息"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
        }
        cap.release()
        return info
    except Exception as e:
        print(f"获取视频信息失败: {e}")
        return None

def get_image_info(image_path):
    """获取图片信息"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        return {
            'width': img.shape[1],
            'height': img.shape[0],
            'channels': img.shape[2] if len(img.shape) > 2 else 1
        }
    except Exception as e:
        print(f"获取图片信息失败: {e}")
        return None

# ============================================================================
# API 路由
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/upload/image', methods=['POST'])
def upload_image():
    """上传人脸图片"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有文件'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400

        if not allowed_file(file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
            return jsonify({'error': '不支持的图片格式'}), 400

        # 保存文件
        filename = generate_unique_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'images', filename)
        file.save(file_path)

        # 获取图片信息
        img_info = get_image_info(file_path)
        file_size = os.path.getsize(file_path)

        # 保存到数据库
        user_id = request.form.get('user_id', 1)  # 默认用户ID为1

        face_image = FaceImage(
            user_id=user_id,
            filename=filename,
            original_filename=file.filename,
            file_path=file_path,
            file_size=file_size,
            width=img_info['width'] if img_info else None,
            height=img_info['height'] if img_info else None
        )
        db.session.add(face_image)
        db.session.commit()

        return jsonify({
            'success': True,
            'id': face_image.id,
            'filename': filename,
            'original_filename': file.filename,
            'url': f'/api/images/{face_image.id}',
            'file_size': file_size,
            'width': img_info['width'] if img_info else None,
            'height': img_info['height'] if img_info else None
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload/video', methods=['POST'])
def upload_video():
    """上传视频文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有文件'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400

        if not allowed_file(file.filename, app.config['ALLOWED_VIDEO_EXTENSIONS']):
            return jsonify({'error': '不支持的视频格式'}), 400

        # 保存文件
        filename = generate_unique_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'videos', filename)
        file.save(file_path)

        # 获取视频信息
        video_info = get_video_info(file_path)
        file_size = os.path.getsize(file_path)

        # 创建缩略图
        thumbnail_path = os.path.join(app.config['UPLOAD_FOLDER'], 'thumbnails', f'{filename}.jpg')
        create_thumbnail(file_path, thumbnail_path)

        # 保存到数据库
        user_id = request.form.get('user_id', 1)

        input_video = InputVideo(
            user_id=user_id,
            filename=filename,
            original_filename=file.filename,
            file_path=file_path,
            file_size=file_size,
            duration=video_info['duration'] if video_info else None,
            width=video_info['width'] if video_info else None,
            height=video_info['height'] if video_info else None,
            fps=video_info['fps'] if video_info else None,
            thumbnail_path=thumbnail_path
        )
        db.session.add(input_video)
        db.session.commit()

        return jsonify({
            'success': True,
            'id': input_video.id,
            'filename': filename,
            'original_filename': file.filename,
            'url': f'/api/videos/{input_video.id}',
            'thumbnail_url': f'/api/thumbnails/{input_video.id}',
            'file_size': file_size,
            'duration': video_info['duration'] if video_info else None,
            'width': video_info['width'] if video_info else None,
            'height': video_info['height'] if video_info else None,
            'fps': video_info['fps'] if video_info else None
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/images', methods=['GET'])
def list_images():
    """获取所有图片列表"""
    try:
        user_id = request.args.get('user_id', 1)
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))

        query = FaceImage.query.filter_by(user_id=user_id, is_active=True)
        total = query.count()
        images = query.order_by(FaceImage.created_at.desc()).offset((page - 1) * per_page).limit(per_page).all()

        return jsonify({
            'success': True,
            'total': total,
            'page': page,
            'per_page': per_page,
            'images': [{
                'id': img.id,
                'filename': img.filename,
                'original_filename': img.original_filename,
                'url': f'/api/images/{img.id}',
                'thumbnail_url': f'/api/thumbnails/{img.id}',
                'file_size': img.file_size,
                'width': img.width,
                'height': img.height,
                'created_at': img.created_at.isoformat()
            } for img in images]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/videos', methods=['GET'])
def list_videos():
    """获取所有视频列表"""
    try:
        user_id = request.args.get('user_id', 1)
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))

        query = InputVideo.query.filter_by(user_id=user_id, is_active=True)
        total = query.count()
        videos = query.order_by(InputVideo.created_at.desc()).offset((page - 1) * per_page).limit(per_page).all()

        return jsonify({
            'success': True,
            'total': total,
            'page': page,
            'per_page': per_page,
            'videos': [{
                'id': vid.id,
                'filename': vid.filename,
                'original_filename': vid.original_filename,
                'url': f'/api/videos/{vid.id}',
                'thumbnail_url': f'/api/thumbnails/{vid.id}',
                'file_size': vid.file_size,
                'duration': vid.duration,
                'width': vid.width,
                'height': vid.height,
                'fps': vid.fps,
                'created_at': vid.created_at.isoformat()
            } for vid in videos]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/images/<int:image_id>', methods=['GET'])
def get_image(image_id):
    """获取图片文件"""
    try:
        image = FaceImage.query.get_or_404(image_id)
        return send_file(image.file_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/videos/<int:video_id>', methods=['GET'])
def get_video(video_id):
    """获取视频文件"""
    try:
        video = InputVideo.query.get_or_404(video_id)
        return send_file(video.file_path, mimetype='video/mp4')
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/thumbnails/<int:file_id>', methods=['GET'])
def get_thumbnail(file_id):
    """获取缩略图（支持图片和视频）"""
    try:
        # 尝试从视频表获取
        video = InputVideo.query.get(file_id)
        if video and video.thumbnail_path:
            return send_file(video.thumbnail_path, mimetype='image/jpeg')

        # 尝试从图片表获取
        image = FaceImage.query.get(file_id)
        if image:
            if image.thumbnail_path:
                return send_file(image.thumbnail_path, mimetype='image/jpeg')
            else:
                return send_file(image.file_path, mimetype='image/jpeg')

        return jsonify({'error': '未找到文件'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/images/<int:image_id>', methods=['DELETE'])
def delete_image(image_id):
    """删除图片"""
    try:
        image = FaceImage.query.get_or_404(image_id)
        image.is_active = False
        db.session.commit()
        return jsonify({'success': True, 'message': '图片已删除'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/videos/<int:video_id>', methods=['DELETE'])
def delete_video(video_id):
    """删除视频"""
    try:
        video = InputVideo.query.get_or_404(video_id)
        video.is_active = False
        db.session.commit()
        return jsonify({'success': True, 'message': '视频已删除'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# ============================================================================
# 启动服务器
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("人脸替换应用后端服务")
    print("=" * 60)
    print(f"上传目录: {app.config['UPLOAD_FOLDER']}")
    print(f"最大文件大小: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024)} MB")
    print(f"数据库: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print("=" * 60)
    print("API 端点:")
    print("  GET  /api/health              - 健康检查")
    print("  POST /api/upload/image        - 上传图片")
    print("  POST /api/upload/video        - 上传视频")
    print("  GET  /api/images              - 获取图片列表")
    print("  GET  /api/videos              - 获取视频列表")
    print("  GET  /api/images/<id>         - 获取图片文件")
    print("  GET  /api/videos/<id>         - 获取视频文件")
    print("  GET  /api/thumbnails/<id>     - 获取缩略图")
    print("  DEL  /api/images/<id>         - 删除图片")
    print("  DEL  /api/videos/<id>         - 删除视频")
    print("=" * 60)

    # 创建表
    with app.app_context():
        db.create_all()
        print("数据库表已创建")

    # 启动服务器
    app.run(host='0.0.0.0', port=5000, debug=True)
