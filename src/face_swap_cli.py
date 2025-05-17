import os
import sys
import time
import cv2
import numpy as np
import dlib
import concurrent.futures
from moviepy.editor import VideoFileClip
import insightface
import logging

# 设置环境变量，解决OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置环境变量以允许使用本地模型
os.environ['INSIGHTFACE_ALLOW_LOCAL_MODEL'] = '1'

# 设置当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
models_folder = os.path.join(base_dir, "models")
output_folder = os.path.join(base_dir, "output_videos")

# 确保文件夹存在
os.makedirs(models_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(base_dir, "logs", f"face_swap_cli_{time.strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("face_swap_cli")

def initialize_models():
    """初始化所有模型"""
    models = {}
    
    # 初始化路径
    cascade_path = os.path.join(models_folder, "haarcascade_frontalface_default.xml")
    predictor_path = os.path.join(models_folder, "shape_predictor_68_face_landmarks.dat")
    inswapper_path = os.path.join(models_folder, "inswapper_128.onnx")
    
    # 初始化OpenCV级联分类器
    if os.path.exists(cascade_path):
        models['face_cascade'] = cv2.CascadeClassifier(cascade_path)
        logger.info("成功加载级联分类器")
    else:
        logger.warning(f"未找到级联分类器文件: {cascade_path}")
        models['face_cascade'] = None
    
    # 初始化dlib检测器
    models['detector'] = dlib.get_frontal_face_detector()
    
    # 初始化dlib特征点预测器
    if os.path.exists(predictor_path):
        try:
            models['predictor'] = dlib.shape_predictor(predictor_path)
            logger.info("成功加载特征点预测模型")
        except Exception as e:
            logger.error(f"加载特征点预测模型失败: {e}")
            models['predictor'] = None
    else:
        logger.warning(f"特征点预测模型文件不存在: {predictor_path}")
        models['predictor'] = None
    
    # 初始化InsightFace模型
    if os.path.exists(inswapper_path):
        try:
            # 设置模型目录
            buffalo_dir = os.path.join(models_folder, 'buffalo_l')
            if not os.path.exists(buffalo_dir):
                os.makedirs(buffalo_dir, exist_ok=True)
                logger.info(f"已创建buffalo_l目录: {buffalo_dir}")
                
            # 初始化face_analyser
            # 强制禁用下载
            os.environ['INSIGHTFACE_ALLOW_LOCAL_MODEL'] = '1'
            
            # 直接使用已安装的模型，不进行下载
            try:
                models['face_analyser'] = insightface.app.FaceAnalysis(
                    name="buffalo_l", 
                    root=models_folder,
                    providers=['CPUExecutionProvider'],
                    allowed_modules=['detection', 'recognition'],
                    download=False
                )
                models['face_analyser'].prepare(ctx_id=0, det_size=(640, 640))
            except Exception as e:
                logger.warning(f"使用download=False初始化失败: {e}")
                
                # 尝试直接加载buffalo_l/onnx目录下的模型文件
                if os.path.exists(os.path.join(buffalo_dir, 'onnx')):
                    logger.info("尝试直接使用已下载的模型文件")
                    models['face_analyser'] = insightface.app.FaceAnalysis(
                        name="buffalo_l", 
                        root=models_folder,
                        providers=['CPUExecutionProvider'],
                        allowed_modules=['detection']
                    )
                    models['face_analyser'].prepare(ctx_id=0, det_size=(640, 640))
                else:
                    logger.warning("无法找到buffalo_l/onnx目录，人脸分析器初始化失败")
                    models['face_analyser'] = None
            
            # 初始化inswapper
            models['inswapper'] = insightface.model_zoo.get_model(
                inswapper_path, 
                providers=['CPUExecutionProvider']
            )
            logger.info(f"成功加载InsightFace ONNX模型: {inswapper_path}")
        except Exception as e:
            logger.error(f"初始化InsightFace模型时出错: {e}")
            import traceback
            traceback.print_exc()
            models['face_analyser'] = None
            models['inswapper'] = None
    else:
        logger.warning(f"InsightFace模型文件不存在: {inswapper_path}")
        models['face_analyser'] = None
        models['inswapper'] = None
    
    return models

def shape_to_np(shape):
    """将dlib的shape转换为numpy数组"""
    coords = np.zeros((shape.num_parts, 2), dtype="int")
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def insightface_face_swap(frame, source_img, models):
    """使用InsightFace进行人脸替换"""
    if models['face_analyser'] is None or models['inswapper'] is None:
        logger.error("InsightFace模型未初始化")
        # 返回原始帧，不进行处理
        return frame
    
    try:
        # 检测原始帧中的人脸
        frame_faces = models['face_analyser'].get(frame)
        if len(frame_faces) == 0:
            logger.warning("未在视频帧中检测到人脸")
            return frame
        
        # 检测源图片中的人脸
        source_faces = models['face_analyser'].get(source_img)
        if len(source_faces) == 0:
            logger.warning("未在源图片中检测到人脸")
            return frame
        
        # 获取源图片中最大的人脸
        source_face = sorted(source_faces, key=lambda x: x.bbox[2] - x.bbox[0], reverse=True)[0]
        
        # 对每个目标人脸进行替换
        result = frame.copy()
        for target_face in frame_faces:
            # 使用InsightFace的模型进行人脸交换
            result = models['inswapper'].get(result, target_face, source_face, paste_back=True)
        
        return result
    except Exception as e:
        logger.error(f"InsightFace人脸替换失败: {e}")
        return frame

def dlib_face_swap(frame, source_img, models):
    """使用dlib进行基本人脸检测"""
    if not models['detector'] or not models['predictor']:
        logger.error("Dlib模型未初始化")
        return frame
    
    try:
        # 使用dlib检测人脸
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = models['detector'](gray_frame)
        
        if len(faces) == 0:
            logger.warning("未在视频帧中检测到人脸")
            return frame
        
        # 检测源图片中的人脸
        gray_source = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
        source_faces = models['detector'](gray_source)
        
        if len(source_faces) == 0:
            logger.warning("未在源图片中检测到人脸")
            return frame
        
        # 在这里，我们只是在检测到的人脸上绘制一个矩形框
        result = frame.copy()
        for face in faces:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 获取面部特征点
            landmarks = models['predictor'](gray_frame, face)
            points = shape_to_np(landmarks)
            
            # 绘制特征点
            for (x, y) in points:
                cv2.circle(result, (x, y), 2, (0, 0, 255), -1)
        
        return result
    except Exception as e:
        logger.error(f"Dlib人脸检测失败: {e}")
        return frame

def process_video(video_path, face_img_path, output_path, models):
    """处理视频的主函数"""
    if not os.path.exists(video_path):
        logger.error(f"视频文件不存在: {video_path}")
        return False
    
    if not os.path.exists(face_img_path):
        logger.error(f"人脸图片不存在: {face_img_path}")
        return False
    
    try:
        # 读取源人脸图片
        source_img = cv2.imread(face_img_path)
        if source_img is None:
            logger.error(f"无法读取源人脸图片: {face_img_path}")
            return False
        
        # 调整源图片大小
        max_height = 720
        h, w = source_img.shape[:2]
        if h > max_height:
            scale = max_height / h
            source_img = cv2.resize(source_img, (int(w * scale), max_height))
        
        # 打开视频
        clip = VideoFileClip(video_path)
        
        # 设置进度回调函数
        total_frames = int(clip.fps * clip.duration)
        processed_frames = 0
        
        def process_frame(frame):
            nonlocal processed_frames
            processed_frames += 1
            if processed_frames % 10 == 0:
                progress = processed_frames / total_frames * 100
                print(f"处理进度: {progress:.2f}% ({processed_frames}/{total_frames})", end="\r")
            
            # 转换为OpenCV格式
            frame_cv = (frame * 255).astype(np.uint8)
            frame_cv = cv2.cvtColor(frame_cv, cv2.COLOR_RGB2BGR)
            
            # 尝试使用InsightFace进行人脸替换
            if models['face_analyser'] is not None and models['inswapper'] is not None:
                result_cv = insightface_face_swap(frame_cv, source_img, models)
            else:
                # 如果InsightFace不可用，则使用dlib进行基本人脸检测
                result_cv = dlib_face_swap(frame_cv, source_img, models)
            
            # 转换回MoviePy格式
            result = cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            return result
        
        # 处理视频
        logger.info(f"开始处理视频: {video_path}")
        new_clip = clip.fl_image(process_frame)
        
        # 保存结果
        logger.info(f"正在保存结果视频到: {output_path}")
        
        # 使用临时文件先保存
        temp_output = os.path.splitext(output_path)[0] + "_temp.mp4"
        try:
            new_clip.write_videofile(temp_output, codec='libx264', audio_codec='aac')
            
            # 检查临时文件的有效性
            if not os.path.exists(temp_output):
                logger.error(f"临时视频文件未能创建: {temp_output}")
                return False
                
            temp_file_size = os.path.getsize(temp_output)
            if temp_file_size < 1000:  # 小于1KB的视频文件很可能是无效的
                logger.error(f"临时视频文件大小异常: {temp_file_size} 字节")
                return False
                
            # 检查通过，重命名为最终文件名
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_output, output_path)
        except Exception as e:
            logger.error(f"保存视频时出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 清理临时文件
            if os.path.exists(temp_output):
                os.remove(temp_output)
            return False
        
        # 清理资源
        clip.close()
        new_clip.close()
        
        logger.info(f"视频处理完成: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"处理视频时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """命令行版本的主函数"""
    # 初始化模型
    logger.info("正在初始化模型...")
    models = initialize_models()
    
    # 获取命令行参数
    if len(sys.argv) < 3:
        print("使用方法: python face_swap_cli.py [视频文件路径] [人脸图片路径] [输出视频路径(可选)]")
        return
    
    video_path = sys.argv[1]
    face_img_path = sys.argv[2]
    
    # 如果没有提供输出路径，则创建一个默认路径
    if len(sys.argv) >= 4:
        output_path = sys.argv[3]
    else:
        # 创建默认输出路径
        filename = os.path.basename(video_path)
        name_without_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f"{name_without_ext}_swapped.mp4")
    
    print(f"视频文件: {video_path}")
    print(f"人脸图片: {face_img_path}")
    print(f"输出文件: {output_path}")
    
    # 处理视频
    success = process_video(video_path, face_img_path, output_path, models)
    
    if success:
        print(f"\n处理完成! 结果已保存到: {output_path}")
    else:
        print("\n处理失败，请查看日志获取详细信息。")

if __name__ == "__main__":
    main() 