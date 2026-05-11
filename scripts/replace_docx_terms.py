from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}
ET.register_namespace("w", W_NS)


@dataclass(frozen=True)
class Replacement:
    source: str
    target: str


REPLACEMENTS = [
    Replacement("working 前端目录", "working front end"),
    Replacement("isolated 演示原型nstration steps", "isolated demonstration steps"),
    Replacement("isolated demo steps", "isolated demonstration steps"),
    Replacement("frontend/main.py", "主程序"),
    Replacement("frontend/face_swap_ui_enhanced.py", "前端UI"),
    Replacement("face_swap_ui_enhanced.py", "前端UI"),
    Replacement("frontend/database_manager.py", "数据库管理模块"),
    Replacement("database_manager.py", "数据库管理模块"),
    Replacement("frontend/face_swap.py", "核心换脸处理模块"),
    Replacement("face_swap.py", "核心换脸处理模块"),
    Replacement("backend-django/db.sqlite3", "本地SQLite数据库文件"),
    Replacement("backend-django", "后端目录"),
    Replacement("frontend", "前端目录"),
    Replacement("output_videos", "输出视频目录"),
    Replacement("logs", "日志目录"),
    Replacement("EnhancedFaceSwapUI", "前端主界面类"),
    Replacement("DatabaseManager", "数据库管理类"),
    Replacement("FaceSwapApp", "换脸核心处理类"),
    Replacement("VideoProcessingThread", "视频处理线程"),
    Replacement("CameraProcessingThread", "摄像头处理线程"),
    Replacement("saveOutputVideoToDatabase", "保存输出视频到数据库"),
    Replacement("ImageUploadThread", "图片上传线程"),
    Replacement("VideoUploadThread", "视频上传线程"),
    Replacement("CircularProgressBar", "环形进度条组件"),
    Replacement("GlowingButton", "发光按钮组件"),
    Replacement("AppMode", "应用模式枚举"),
    Replacement("local_path", "本地路径字段"),
    Replacement("original_filename", "原始文件名"),
    Replacement("file_size", "文件大小"),
    Replacement("duration", "时长"),
    Replacement("fps", "帧率"),
    Replacement("FPS", "帧/秒"),
    Replacement("width", "宽度"),
    Replacement("height", "高度"),
    Replacement("filename", "文件名"),
    Replacement("processing_method", "处理方法"),
    Replacement("processing_time", "处理时间"),
    Replacement("status", "状态"),
    Replacement("progress", "进度"),
    Replacement("task_type", "任务类型"),
    Replacement("processing_params", "处理参数"),
    Replacement("processing_tasks", "处理任务表"),
    Replacement("FaceImage", "人脸图片表"),
    Replacement("InputVideo", "输入视频表"),
    Replacement("OutputVideo", "输出视频表"),
    Replacement("ProcessingTask", "处理任务表"),
    Replacement("input_video", "输入视频"),
    Replacement("face_image", "人脸图片"),
    Replacement("original_app", "核心换脸处理对象"),
    Replacement("loadFaceImagesFromLocal", "从本地加载人脸图片"),
    Replacement("loadVideoFiles", "加载本地视频文件"),
    Replacement("loadFaceImages", "加载人脸图片"),
    Replacement("loadVideos", "加载视频素材"),
    Replacement("startProcessing", "开始处理"),
    Replacement("processingFinished", "处理完成回调"),
    Replacement("loadProcessedVideo", "加载处理后视频"),
    Replacement("playWithOpenCV", "使用OpenCV播放"),
    Replacement("togglePlayback", "切换播放状态"),
    Replacement("changePlaybackSpeed", "调整播放速度"),
    Replacement("switchMode", "模式切换"),
    Replacement("process_frame_traditional", "传统方法单帧处理"),
    Replacement("advanced_face_swap", "高级换脸处理"),
    Replacement("simple_face_swap", "基础换脸处理"),
    Replacement("seamlessClone", "无缝融合函数"),
    Replacement("progress_signal", "进度信号"),
    Replacement("status_signal", "状态信号"),
    Replacement("finished_signal", "完成信号"),
    Replacement("error_signal", "错误信号"),
    Replacement("benchmark", "基准测试"),
    Replacement("纯算法 demo", "纯算法演示原型"),
    Replacement("纯算法 演示原型", "纯算法演示原型"),
    Replacement("self.face_analyser.get", "人脸分析器的获取方法"),
    Replacement(
        "self.inswapper.get(result, face, source_face, paste_back=True)",
        "换脸推理方法",
    ),
    Replacement("traditional / inswapper", "传统方法 / inswapper"),
]


def paragraph_text(paragraph: ET.Element) -> str:
    return "".join(node.text or "" for node in paragraph.findall(".//w:t", NS))


def redistribute_text(paragraph: ET.Element, new_text: str) -> None:
    text_nodes = paragraph.findall(".//w:t", NS)
    if not text_nodes:
        return

    remaining = new_text
    for index, node in enumerate(text_nodes):
        if index == len(text_nodes) - 1:
            node.text = remaining
            remaining = ""
            continue

        current = node.text or ""
        take = min(len(current), len(remaining))
        node.text = remaining[:take]
        remaining = remaining[take:]

    if remaining:
        text_nodes[-1].text = (text_nodes[-1].text or "") + remaining


def replace_in_document_xml(xml_bytes: bytes) -> tuple[bytes, list[tuple[str, str, str]]]:
    root = ET.fromstring(xml_bytes)
    changes: list[tuple[str, str, str]] = []

    for paragraph in root.findall(".//w:p", NS):
        old_text = paragraph_text(paragraph)
        if not old_text:
            continue

        new_text = old_text
        for item in REPLACEMENTS:
            if item.source in new_text:
                new_text = new_text.replace(item.source, item.target)

        if new_text != old_text:
            redistribute_text(paragraph, new_text)
            changes.append((old_text, new_text, new_text[:80]))

    return ET.tostring(root, encoding="utf-8", xml_declaration=True), changes


def inspect_document(docx_path: Path) -> None:
    with zipfile.ZipFile(docx_path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))

    for index, paragraph in enumerate(root.findall(".//w:p", NS), start=1):
        text = paragraph_text(paragraph).strip()
        if text and any(item.source in text for item in REPLACEMENTS):
            print(f"[{index}] {text}")


def replace_document(docx_path: Path, create_backup: bool) -> None:
    if create_backup:
        backup_path = docx_path.with_suffix(docx_path.suffix + ".bak")
        shutil.copy2(docx_path, backup_path)
        print(f"Backup: {backup_path}")

    with zipfile.ZipFile(docx_path, "r") as src:
        file_map = {name: src.read(name) for name in src.namelist()}

    new_xml, changes = replace_in_document_xml(file_map["word/document.xml"])
    file_map["word/document.xml"] = new_xml

    temp_path = docx_path.with_suffix(docx_path.suffix + ".tmp")
    with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as dst:
        for name, data in file_map.items():
            dst.writestr(name, data)

    try:
        temp_path.replace(docx_path)
        print(f"Updated: {docx_path}")
        print(f"Changed paragraphs: {len(changes)}")
    except PermissionError:
        temp_path.unlink(missing_ok=True)
        replace_via_word_com(docx_path)
        print(f"Updated via Word COM: {docx_path}")
        print("Changed content by replacement table because the file was locked by Word.")


def replace_via_word_com(docx_path: Path) -> None:
    import pythoncom
    import win32com.client

    word = None
    doc = None
    created_word = False
    opened_here = False

    pythoncom.CoInitialize()
    try:
        try:
            word = win32com.client.GetActiveObject("Word.Application")
        except Exception:
            word = win32com.client.DispatchEx("Word.Application")
            word.Visible = False
            created_word = True

        full_path = str(docx_path.resolve()).lower()
        for candidate in word.Documents:
            if str(candidate.FullName).lower() == full_path:
                doc = candidate
                break

        if doc is None:
            doc = word.Documents.Open(str(docx_path.resolve()), ReadOnly=False)
            opened_here = True

        content_range = doc.Content
        for item in REPLACEMENTS:
            find = content_range.Find
            find.ClearFormatting()
            find.Replacement.ClearFormatting()
            find.Execute(
                FindText=item.source,
                MatchCase=True,
                MatchWholeWord=False,
                MatchWildcards=False,
                MatchSoundsLike=False,
                MatchAllWordForms=False,
                Forward=True,
                Wrap=1,
                Format=False,
                ReplaceWith=item.target,
                Replace=2,
            )

        doc.Save()
    finally:
        if doc is not None and opened_here:
            doc.Close()
        if word is not None and created_word:
            word.Quit()
        pythoncom.CoUninitialize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("docx_path", type=Path)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    if args.inspect == args.replace:
        raise SystemExit("Use exactly one of --inspect or --replace.")

    if args.inspect:
        inspect_document(args.docx_path)
        return

    replace_document(args.docx_path, create_backup=not args.no_backup)


if __name__ == "__main__":
    main()
