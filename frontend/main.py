#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
人脸替换应用 - 前端主入口
"""
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# 导入主窗口（从复制的文件）
from face_swap_ui_enhanced import EnhancedFaceSwapUI

def main():
    """主函数"""
    # 启用高DPI支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("人脸替换应用")
    app.setOrganizationName("FaceSwap")
    app.setStyle("Fusion")

    # 暗色科技感主题样式表 - 增强版
    style_sheet = """
    * {
        font-family: 'Segoe UI', 'Microsoft YaHei UI', sans-serif;
    }

    QMainWindow {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                    stop:0 #0f1419, stop:0.5 #1a1f2e, stop:1 #0f1419);
    }

    QWidget {
        background-color: transparent;
        color: #e0e6ed;
    }

    QPushButton {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 #4a5568, stop:1 #2d3748);
        color: #e0e6ed;
        border: 2px solid #63b3ed;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 14px;
    }

    QPushButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 #667eea, stop:0.5 #764ba2, stop:1 #667eea);
        color: #ffffff;
        border: 2px solid #90cdf4;
    }

    QPushButton:pressed {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 #553c9a, stop:1 #44337a);
    }

    QPushButton:checked {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 #667eea, stop:0.5 #764ba2, stop:1 #667eea);
        color: #ffffff;
        border: 2px solid #90cdf4;
    }

    QGroupBox {
        color: #90cdf4;
        font-weight: 700;
        font-size: 15px;
        border: 2px solid #4299e1;
        border-radius: 12px;
        margin-top: 14px;
        padding-top: 22px;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                    stop:0 rgba(26, 32, 44, 0.7),
                                    stop:1 rgba(45, 55, 72, 0.7));
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 18px;
        padding: 4px 12px;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 #4299e1, stop:1 #3182ce);
        border-radius: 6px;
        color: #ffffff;
    }

    QLabel {
        color: #e0e6ed;
        font-size: 14px;
    }

    QProgressBar {
        border: 2px solid #4299e1;
        border-radius: 8px;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 rgba(26, 32, 44, 0.8),
                                    stop:1 rgba(45, 55, 72, 0.8));
        text-align: center;
        height: 24px;
        color: #ffffff;
    }

    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 #667eea, stop:0.3 #764ba2, stop:0.7 #667eea, stop:1 #4299e1);
        border-radius: 6px;
    }

    QSlider::groove:horizontal {
        border: 2px solid #4299e1;
        height: 8px;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 #2d3748, stop:1 #4a5568);
        border-radius: 4px;
    }

    QSlider::handle:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 #90cdf4, stop:1 #4299e1);
        border: 2px solid #63b3ed;
        width: 20px;
        height: 20px;
        margin: -7px 0;
        border-radius: 10px;
    }

    QSlider::handle:horizontal:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 #bee3f8, stop:1 #90cdf4);
        border: 2px solid #90cdf4;
    }

    QListWidget {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                    stop:0 rgba(26, 32, 44, 0.8),
                                    stop:1 rgba(45, 55, 72, 0.8));
        border: 2px solid #4299e1;
        border-radius: 8px;
        padding: 6px;
        outline: none;
    }

    QListWidget::item {
        padding: 8px 10px;
        border-radius: 6px;
        margin: 3px;
        color: #e0e6ed;
        background: rgba(45, 55, 72, 0.3);
    }

    QListWidget::item:selected {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 #667eea, stop:0.5 #764ba2, stop:1 #667eea);
        color: #ffffff;
        border: 1px solid #90cdf4;
    }

    QListWidget::item:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 rgba(66, 153, 225, 0.4),
                                    stop:1 rgba(102, 126, 234, 0.4));
    }

    QComboBox {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 rgba(45, 55, 72, 0.8),
                                    stop:1 rgba(26, 32, 44, 0.8));
        border: 2px solid #4299e1;
        border-radius: 6px;
        padding: 6px 12px;
        color: #e0e6ed;
    }

    QComboBox:hover {
        border: 2px solid #63b3ed;
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 rgba(66, 153, 225, 0.6),
                                    stop:1 rgba(45, 55, 72, 0.6));
    }

    QComboBox::drop-down {
        border: none;
        width: 24px;
    }

    QComboBox::down-arrow {
        image: none;
        border: 6px solid transparent;
        border-top-color: #90cdf4;
        margin-right: 6px;
    }

    QComboBox QAbstractItemView {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                    stop:0 rgba(26, 32, 44, 0.95),
                                    stop:1 rgba(45, 55, 72, 0.95));
        border: 2px solid #4299e1;
        color: #e0e6ed;
        selection-background-color: #667eea;
        selection-color: #ffffff;
    }

    QLineEdit {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 rgba(45, 55, 72, 0.8),
                                    stop:1 rgba(26, 32, 44, 0.8));
        border: 2px solid #4299e1;
        border-radius: 6px;
        padding: 6px 12px;
        color: #e0e6ed;
    }

    QLineEdit:focus {
        border: 2px solid #63b3ed;
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 rgba(66, 153, 225, 0.5),
                                    stop:1 rgba(45, 55, 72, 0.5));
    }

    QScrollBar:vertical {
        background: rgba(26, 32, 44, 0.6);
        width: 12px;
        border-radius: 6px;
    }

    QScrollBar::handle:vertical {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 #4a5568, stop:1 #2d3748);
        border-radius: 6px;
        min-height: 30px;
        border: 1px solid #4299e1;
    }

    QScrollBar::handle:vertical:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 #667eea, stop:1 #4299e1);
    }

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }

    QScrollBar:horizontal {
        background: rgba(26, 32, 44, 0.6);
        height: 12px;
        border-radius: 6px;
    }

    QScrollBar::handle:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 #4a5568, stop:1 #2d3748);
        border-radius: 6px;
        min-width: 30px;
        border: 1px solid #4299e1;
    }

    QScrollBar::handle:horizontal:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 #667eea, stop:1 #4299e1);
    }

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
    }

    QTabWidget::pane {
        border: 2px solid #4299e1;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                    stop:0 rgba(26, 32, 44, 0.6),
                                    stop:1 rgba(45, 55, 72, 0.6));
        border-radius: 8px;
    }

    QTabBar::tab {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 rgba(45, 55, 72, 0.8),
                                    stop:1 rgba(26, 32, 44, 0.8));
        color: #90cdf4;
        padding: 10px 18px;
        border: 2px solid #4299e1;
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        margin-right: 3px;
        font-weight: 600;
    }

    QTabBar::tab:selected {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 #667eea, stop:1 #4299e1);
        color: #ffffff;
        border-bottom: 2px solid #667eea;
    }

    QTabBar::tab:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 #667eea, stop:1 #4299e1);
        color: #ffffff;
    }

    QCheckBox {
        color: #e0e6ed;
    }

    QCheckBox::indicator {
        width: 20px;
        height: 20px;
        border: 2px solid #4299e1;
        border-radius: 4px;
        background: rgba(45, 55, 72, 0.8);
    }

    QCheckBox::indicator:checked {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 #667eea, stop:1 #4299e1);
        border-color: #63b3ed;
    }

    QCheckBox::indicator:hover {
        border-color: #63b3ed;
    }

    QRadioButton {
        color: #e0e6ed;
    }

    QRadioButton::indicator {
        width: 20px;
        height: 20px;
        border: 2px solid #4299e1;
        border-radius: 10px;
        background: rgba(45, 55, 72, 0.8);
    }

    QRadioButton::indicator:checked {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 #667eea, stop:1 #4299e1);
        border-color: #63b3ed;
    }

    QRadioButton::indicator:hover {
        border-color: #63b3ed;
    }

    QStatusBar {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 rgba(26, 32, 44, 0.9),
                                    stop:1 rgba(45, 55, 72, 0.9));
        color: #90cdf4;
        border-top: 2px solid #4299e1;
    }
    """

    app.setStyleSheet(style_sheet)
    print("已应用暗色科技感主题")

    # 创建主窗口
    window = EnhancedFaceSwapUI()
    window.show()

    # 退出时清理
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
