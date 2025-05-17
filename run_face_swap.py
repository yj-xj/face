#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI人脸替换系统启动脚本
"""

import os
import sys
import logging
from datetime import datetime

# 设置环境变量，解决OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 解决Tcl/Tk版本冲突问题
os.environ['TCL_IGNORE_VERSION_CHECK'] = '1'

# 设置日志
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"app_launcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("app_launcher")

def main():
    """主函数"""
    try:
        # 添加当前目录和src目录到系统路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(current_dir, "src")
        if src_dir not in sys.path:
            sys.path.append(src_dir)
            
        # 默认尝试启动增强版UI
        try:
            logger.info("尝试导入增强版UI...")
            from src.face_swap_ui_enhanced import main as enhanced_main
            logger.info("启动增强版UI...")
            enhanced_main()
        except ImportError as e:
            logger.error(f"无法导入增强版UI: {e}")
            logger.info("启动基本版UI...")
            from src.face_swap import main as basic_main
            basic_main()
    except Exception as e:
        logger.error(f"启动增强版UI时出错: {e}")
        import traceback
        traceback.print_exc()
        logger.info("启动基本版UI...")
        try:
            from src.face_swap import main as basic_main
            basic_main()
        except Exception as e2:
            logger.error(f"启动基本版UI时出错: {e2}")
            traceback.print_exc()
            print("无法启动任何UI版本。请检查日志文件以获取更多信息。")
            input("按Enter键退出...")
        
if __name__ == "__main__":
    main() 