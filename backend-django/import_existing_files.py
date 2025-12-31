#!/usr/bin/env python
"""
将现有文件导入到Django数据库中
"""
import os
import sys
import django
import cv2
from PIL import Image as PILImage

# 设置Django环境
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_swap.settings')
django.setup()

from django.contrib.auth.models import User
from core.models import FaceImage, InputVideo, OutputVideo


def get_or_create_default_user():
    """获取或创建默认用户"""
    user, created = User.objects.get_or_create(
        username='default_user',
        defaults={'email': 'default@example.com'}
    )
    return user


def import_face_images(user, input_faces_dir):
    """导入人脸图片"""
    print(f"\n=== 开始导入人脸图片: {input_faces_dir} ===")

    if not os.path.exists(input_faces_dir):
        print(f"目录不存在: {input_faces_dir}")
        return

    files = [f for f in os.listdir(input_faces_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    print(f"找到 {len(files)} 个图片文件")

    for filename in files:
        file_path = os.path.join(input_faces_dir, filename)

        try:
            # 检查是否已存在
            if FaceImage.objects.filter(original_filename=filename).exists():
                print(f"跳过已存在的图片: {filename}")
                continue

            # 读取图片获取信息
            with PILImage.open(file_path) as img:
                width, height = img.size
                file_size = os.path.getsize(file_path)

            # 生成唯一filename
            import uuid
            ext = os.path.splitext(filename)[1]
            unique_filename = f"{uuid.uuid4().hex}{ext}"

            # 创建数据库记录
            with open(file_path, 'rb') as f:
                from django.core.files.base import ContentFile
                face_image = FaceImage(
                    user=user,
                    filename=unique_filename,
                    original_filename=filename,
                    file_size=file_size,
                    width=width,
                    height=height,
                    is_active=True
                )
                face_image.file.save(filename, ContentFile(f.read()), save=True)

            print(f"[OK] 导入成功: {filename} ({width}x{height})")

        except Exception as e:
            print(f"[FAIL] 导入失败 {filename}: {e}")

    print(f"=== 人脸图片导入完成 ===")


def import_input_videos(user, input_videos_dir):
    """导入输入视频"""
    print(f"\n=== 开始导入输入视频: {input_videos_dir} ===")

    if not os.path.exists(input_videos_dir):
        print(f"目录不存在: {input_videos_dir}")
        return

    files = [f for f in os.listdir(input_videos_dir)
             if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'))]

    print(f"找到 {len(files)} 个视频文件")

    for filename in files:
        file_path = os.path.join(input_videos_dir, filename)

        try:
            # 检查是否已存在
            if InputVideo.objects.filter(original_filename=filename).exists():
                print(f"跳过已存在的视频: {filename}")
                continue

            # 读取视频获取信息
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(f"无法打开视频: {filename}")
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            file_size = os.path.getsize(file_path)
            cap.release()

            # 生成唯一filename
            import uuid
            ext = os.path.splitext(filename)[1]
            unique_filename = f"{uuid.uuid4().hex}{ext}"

            # 创建数据库记录
            with open(file_path, 'rb') as f:
                from django.core.files.base import ContentFile
                input_video = InputVideo(
                    user=user,
                    filename=unique_filename,
                    original_filename=filename,
                    file_size=file_size,
                    duration=duration,
                    width=width,
                    height=height,
                    fps=fps,
                    is_active=True
                )
                input_video.file.save(filename, ContentFile(f.read()), save=True)

            print(f"[OK] 导入成功: {filename} ({width}x{height} @{fps:.2f}fps {duration:.1f}s)")

        except Exception as e:
            print(f"[FAIL] 导入失败 {filename}: {e}")

    print(f"=== 输入视频导入完成 ===")


def import_output_videos(user, output_videos_dir):
    """导入输出视频"""
    print(f"\n=== 开始导入输出视频: {output_videos_dir} ===")

    if not os.path.exists(output_videos_dir):
        print(f"目录不存在: {output_videos_dir}")
        return

    # 首先需要有一个输入视频和人脸图片
    input_video = InputVideo.objects.filter(user=user).first()
    face_image = FaceImage.objects.filter(user=user).first()

    if not input_video or not face_image:
        print("警告: 缺少输入视频或人脸图片，无法导入输出视频")
        print(f"  输入视频: {input_video}")
        print(f"  人脸图片: {face_image}")
        return

    files = [f for f in os.listdir(output_videos_dir)
             if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'))]

    print(f"找到 {len(files)} 个视频文件")
    print(f"关联输入视频: {input_video.original_filename}")
    print(f"关联人脸图片: {face_image.original_filename}")

    for filename in files:
        file_path = os.path.join(output_videos_dir, filename)

        try:
            # 检查是否已存在
            if OutputVideo.objects.filter(filename=filename).exists():
                print(f"跳过已存在的视频: {filename}")
                continue

            # 读取视频获取信息
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(f"无法打开视频: {filename}")
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            file_size = os.path.getsize(file_path)
            cap.release()

            # 创建输出视频记录
            with open(file_path, 'rb') as f:
                from django.core.files.base import ContentFile
                output_video = OutputVideo(
                    user=user,
                    input_video=input_video,
                    face_image=face_image,
                    filename=filename,
                    file_size=file_size,
                    duration=duration,
                    width=width,
                    height=height,
                    fps=fps,
                    processing_method='inswapper',
                    processing_time=duration * 0.5,  # 估计处理时间
                    status='completed',
                    progress=100
                )
                output_video.file.save(filename, ContentFile(f.read()), save=True)

            print(f"[OK] 导入成功: {filename}")

        except Exception as e:
            print(f"[FAIL] 导入失败 {filename}: {e}")

    print(f"=== 输出视频导入完成 ===")


def main():
    """主函数"""
    print("=" * 60)
    print("开始导入现有文件到数据库")
    print("=" * 60)

    # 获取用户
    user = get_or_create_default_user()
    print(f"使用用户: {user.username} (ID: {user.id})")

    # 定义目录路径
    base_dir = r"E:\face"
    input_faces_dir = os.path.join(base_dir, "data", "input_faces")
    input_videos_dir = os.path.join(base_dir, "data", "input_videos")
    output_videos_dir = os.path.join(base_dir, "output_videos")

    # 导入各类文件
    import_face_images(user, input_faces_dir)
    import_input_videos(user, input_videos_dir)
    import_output_videos(user, output_videos_dir)

    # 统计结果
    print("\n" + "=" * 60)
    print("导入完成统计:")
    print(f"  人脸图片: {FaceImage.objects.filter(user=user).count()} 条")
    print(f"  输入视频: {InputVideo.objects.filter(user=user).count()} 条")
    print(f"  输出视频: {OutputVideo.objects.filter(user=user).count()} 条")
    print("=" * 60)


if __name__ == '__main__':
    main()
