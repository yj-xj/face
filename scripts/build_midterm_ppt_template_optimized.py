# -*- coding: utf-8 -*-
from __future__ import annotations

import shutil
from pathlib import Path
from urllib import request

import win32com.client


ROOT = Path(__file__).resolve().parents[1]
OUT_PPTX = ROOT / "midterm_face_swap_defense_template_optimized_v2.pptx"
OUT_PDF = ROOT / "midterm_face_swap_defense_template_optimized_v2.pdf"
PREVIEW_DIR = ROOT / "debug" / "midterm_face_swap_defense_template_preview_v2"
TEMP_DIR = ROOT / "debug" / "template_optimized_temp"
ICON_CACHE_DIR = ROOT / "thesis_assets_generated" / "icons"
SIMPLE_ICON_BASE = "https://cdn.jsdelivr.net/npm/simple-icons/icons"

SLIDE_W = 960
SLIDE_H = 540

PP_LAYOUT_BLANK = 12
PPTX_FORMAT = 24
PDF_FORMAT = 32
TEXT_HORIZONTAL = 1
SHAPE_RECT = 1
SHAPE_ROUND = 5
SHAPE_OVAL = 9
ALIGN_LEFT = 1
ALIGN_CENTER = 2


COLORS = {
    "blue": (11, 90, 134),
    "blue_dark": (18, 52, 83),
    "ink": (33, 58, 88),
    "muted": (92, 109, 129),
    "line": (143, 177, 198),
    "white": (255, 255, 255),
    "paper": (249, 251, 253),
    "blue_soft": (239, 247, 252),
    "blue_lite": (229, 240, 248),
    "gold_soft": (255, 245, 215),
    "green_soft": (237, 246, 236),
    "red_soft": (253, 238, 238),
    "gray_block": (66, 66, 68),
}


ASSETS = {
    "bg": ROOT / "debug" / "template_extract" / "image1.png",
    "logo": ROOT / "debug" / "sdu_logo.png",
    "ui_main": ROOT / "thesis_assets_generated" / "ui_main.png",
    "pair_a": ROOT / "thesis_assets_generated" / "pair_57272a8d42.png",
    "website_full": ROOT / "thesis_assets_generated" / "website_home.png",
    "architecture": ROOT / "thesis_assets_generated" / "two_row_9f04ed605a.png",
    "flow_video": ROOT / "thesis_assets_generated" / "flow_27fe4702f4.png",
    "flow_camera": ROOT / "thesis_assets_generated" / "flow_1c04cb8b48.png",
    "system_screenshot": ROOT / "debug" / "doc_media_preview" / "image8.png",
    "thesis_screenshot": ROOT / "debug" / "doc_media_preview" / "image16.png",
    "traditional_input_frame": ROOT / "thesis_assets_generated" / "input_video_frame.png",
    "traditional_output_frame": ROOT / "thesis_assets_generated" / "output_video_frame.png",
    "ai_sample_video": ROOT / "output_videos" / "Video_1728365311307_face_swap.mp4",
    "ai_fallback_frame": ROOT / "image" / "face_swap1.png",
}

ICON_SLUGS = {
    "qt": "qt",
    "python": "python",
    "opencv": "opencv",
    "django": "django",
    "sqlite": "sqlite",
    "onnx": "onnx",
    "json": "json",
    "openapi": "openapiinitiative",
}


def rgb(color: tuple[int, int, int]) -> int:
    r, g, b = color
    return r + (g << 8) + (b << 16)


def safe_unlink(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        path.unlink()
        return True
    except PermissionError:
        return False


def fit_box(img_w: int, img_h: int, box_w: float, box_h: float) -> tuple[float, float]:
    scale = min(box_w / img_w, box_h / img_h)
    return img_w * scale, img_h * scale


def first_existing_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def extract_video_middle_frame(video_path: Path, output_path: Path) -> Path | None:
    try:
        import cv2
    except Exception:
        return None

    if not video_path.exists():
        return None

    cap = None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap or not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_idx = max(0, total_frames // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), frame):
            return None
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path
        return None
    except Exception:
        return None
    finally:
        if cap is not None:
            cap.release()


def prepare_temp_assets() -> dict[str, object]:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    website_top = first_existing_path([ASSETS["website_full"], ASSETS["ui_main"]])
    icons = prepare_icon_assets()

    system_screenshot = first_existing_path([ASSETS["system_screenshot"], ASSETS["ui_main"]])
    thesis_screenshot = first_existing_path([ASSETS["thesis_screenshot"], ASSETS["pair_a"]])
    traditional_input = first_existing_path([ASSETS["traditional_input_frame"], ASSETS["pair_a"]])
    traditional_output = first_existing_path([ASSETS["traditional_output_frame"], ASSETS["pair_a"]])

    ai_sample_frame = extract_video_middle_frame(ASSETS["ai_sample_video"], TEMP_DIR / "ai_sample_frame.png")
    ai_sample_fallback = False
    if ai_sample_frame is None:
        ai_sample_fallback = True
        ai_sample_frame = first_existing_path(
            [ASSETS["ai_fallback_frame"], ASSETS["pair_a"], ASSETS["traditional_output_frame"]]
        )

    if ai_sample_frame is None:
        ai_sample_note = "AI 样例素材缺失，本页使用占位说明。"
    elif ai_sample_fallback:
        ai_sample_note = "AI 样例帧抽取失败，当前展示历史运行样例。"
    else:
        ai_sample_note = "AI 样例来自 output_videos 同场景输出视频的中间帧。"

    return {
        "website_top": website_top,
        "icons": icons,
        "system_screenshot": system_screenshot,
        "thesis_screenshot": thesis_screenshot,
        "traditional_input_frame": traditional_input,
        "traditional_output_frame": traditional_output,
        "ai_sample_frame": ai_sample_frame,
        "ai_sample_note": ai_sample_note,
    }


def download_icon(slug: str) -> Path | None:
    ICON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = ICON_CACHE_DIR / f"{slug}.svg"
    if target.exists() and target.stat().st_size > 0:
        return target
    url = f"{SIMPLE_ICON_BASE}/{slug}.svg"
    try:
        with request.urlopen(url, timeout=8) as response:
            data = response.read()
        if not data:
            return None
        target.write_bytes(data)
        return target
    except Exception:
        return None


def prepare_icon_assets() -> dict[str, Path | None]:
    icons: dict[str, Path | None] = {}
    for key, slug in ICON_SLUGS.items():
        icons[key] = download_icon(slug)
    return icons


def set_text(shape, text: str, font_size: float, color: tuple[int, int, int], *, bold: bool = False, align: int = ALIGN_LEFT, font_name: str = "Microsoft YaHei") -> None:
    tf = shape.TextFrame
    tf.TextRange.Text = text
    tf.MarginLeft = 5
    tf.MarginRight = 5
    tf.MarginTop = 4
    tf.MarginBottom = 4
    tf.WordWrap = True
    tr = tf.TextRange
    tr.Font.Name = font_name
    tr.Font.Size = font_size
    tr.Font.Bold = -1 if bold else 0
    tr.Font.Color.RGB = rgb(color)
    tr.ParagraphFormat.Alignment = align


def add_textbox(slide, left: float, top: float, width: float, height: float, text: str, font_size: float, color: tuple[int, int, int], *, bold: bool = False, align: int = ALIGN_LEFT, font_name: str = "Microsoft YaHei"):
    shape = slide.Shapes.AddTextbox(TEXT_HORIZONTAL, left, top, width, height)
    shape.Line.Visible = 0
    shape.Fill.Visible = 0
    set_text(shape, text, font_size, color, bold=bold, align=align, font_name=font_name)
    return shape


def add_shape(slide, shape_type: int, left: float, top: float, width: float, height: float, fill: tuple[int, int, int], *, line: tuple[int, int, int] | None = None, line_weight: float = 1.0, transparency: float = 0.0):
    shape = slide.Shapes.AddShape(shape_type, left, top, width, height)
    shape.Fill.Visible = -1
    shape.Fill.Solid()
    shape.Fill.ForeColor.RGB = rgb(fill)
    shape.Fill.Transparency = transparency
    if line is None:
        shape.Line.Visible = 0
    else:
        shape.Line.Visible = -1
        shape.Line.ForeColor.RGB = rgb(line)
        shape.Line.Weight = line_weight
    return shape


def add_round_box(slide, left: float, top: float, width: float, height: float, fill: tuple[int, int, int], *, line: tuple[int, int, int] | None = None, line_weight: float = 1.0, transparency: float = 0.0):
    return add_shape(slide, SHAPE_ROUND, left, top, width, height, fill, line=line, line_weight=line_weight, transparency=transparency)


def add_rect(slide, left: float, top: float, width: float, height: float, fill: tuple[int, int, int], *, line: tuple[int, int, int] | None = None, line_weight: float = 1.0, transparency: float = 0.0):
    return add_shape(slide, SHAPE_RECT, left, top, width, height, fill, line=line, line_weight=line_weight, transparency=transparency)


def add_circle(slide, left: float, top: float, width: float, height: float, fill: tuple[int, int, int], *, line: tuple[int, int, int] | None = None, line_weight: float = 1.0, transparency: float = 0.0):
    return add_shape(slide, SHAPE_OVAL, left, top, width, height, fill, line=line, line_weight=line_weight, transparency=transparency)


def add_picture_fit(slide, path: Path, left: float, top: float, width: float, height: float):
    pic = slide.Shapes.AddPicture(str(path), False, True, left, top, -1, -1)
    img_w, img_h = float(pic.Width), float(pic.Height)
    if img_w <= 0 or img_h <= 0:
        pic.Width = width
        pic.Height = height
        pic.Left = left
        pic.Top = top
        return pic
    new_w, new_h = fit_box(img_w, img_h, width, height)
    pic.Width = new_w
    pic.Height = new_h
    pic.Left = left + (width - new_w) / 2
    pic.Top = top + (height - new_h) / 2
    return pic


def add_image_or_placeholder(
    slide,
    path: Path | None,
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    placeholder: str = "素材缺失",
):
    if path and path.exists() and path.stat().st_size > 0:
        try:
            return add_picture_fit(slide, path, left, top, width, height)
        except Exception:
            pass

    add_round_box(slide, left, top, width, height, COLORS["paper"], line=COLORS["line"], line_weight=0.9)
    add_textbox(slide, left + 8, top + (height / 2) - 10, width - 16, 20, placeholder, 10.5, COLORS["muted"], align=ALIGN_CENTER)
    return None


def add_picture_size(slide, path: Path, left: float, top: float, width: float, height: float):
    return slide.Shapes.AddPicture(str(path), False, True, left, top, width, height)


def add_line(slide, x1: float, y1: float, x2: float, y2: float, color: tuple[int, int, int], *, weight: float = 1.5):
    line = slide.Shapes.AddLine(x1, y1, x2, y2)
    line.Line.ForeColor.RGB = rgb(color)
    line.Line.Weight = weight
    return line


def add_arrow_line(
    slide,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: tuple[int, int, int],
    *,
    weight: float = 1.5,
    bidirectional: bool = False,
):
    line = add_line(slide, x1, y1, x2, y2, color, weight=weight)
    try:
        line.Line.EndArrowheadStyle = 3
        if bidirectional:
            line.Line.BeginArrowheadStyle = 3
    except Exception:
        # Some Office builds may not expose arrow style constants.
        pass
    return line


def add_icon_or_fallback(
    slide,
    icon_path: Path | None,
    left: float,
    top: float,
    size: float,
    fallback_text: str,
    *,
    fill: tuple[int, int, int] = COLORS["blue_lite"],
) -> None:
    if icon_path is not None and icon_path.exists():
        try:
            add_picture_size(slide, icon_path, left, top, size, size)
            return
        except Exception:
            pass
    badge = add_round_box(slide, left, top, size, size, fill, line=COLORS["line"], line_weight=0.8)
    set_text(badge, fallback_text, 9.2, COLORS["blue_dark"], bold=True, align=ALIGN_CENTER, font_name="Segoe UI")


def add_tech_item(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
    icon_path: Path | None,
    fallback_text: str,
    *,
    fill: tuple[int, int, int] = COLORS["white"],
    accent: tuple[int, int, int] = COLORS["blue"],
) -> None:
    add_round_box(slide, left, top, width, height, fill, line=COLORS["line"], line_weight=0.9)
    add_rect(slide, left, top, width, 4, accent, line=None)
    add_icon_or_fallback(slide, icon_path, left + 10, top + 12, 24, fallback_text, fill=COLORS["blue_soft"])
    add_textbox(slide, left + 42, top + 11, width - 52, 20, title, 11.8, COLORS["blue_dark"], bold=True)
    add_textbox(slide, left + 12, top + 35, width - 24, 20, subtitle, 9.6, COLORS["muted"])


def add_bg(slide) -> None:
    slide.FollowMasterBackground = False
    fill = slide.Background.Fill
    fill.Visible = -1
    fill.Solid()
    fill.ForeColor.RGB = rgb(COLORS["white"])
    add_picture_size(slide, ASSETS["bg"], 0, 0, SLIDE_W, SLIDE_H)
    add_rect(slide, 0, 0, SLIDE_W, 8, COLORS["blue"], line=None)


def add_logo_badge(slide, *, cover: bool = False) -> None:
    if cover:
        add_picture_size(slide, ASSETS["logo"], 38, 28, 170, 52)
    else:
        add_picture_size(slide, ASSETS["logo"], 798, 18, 132, 41)


def add_cover_decor(slide) -> None:
    add_rect(slide, 0, 0, SLIDE_W, 10, COLORS["blue"], line=None)
    add_rect(slide, 0, 508, 750, 10, COLORS["blue"], line=None)
    for x, y, c in [(22, 48, COLORS["gray_block"]), (36, 56, COLORS["blue"]), (50, 64, COLORS["gray_block"]), (36, 72, COLORS["gray_block"])]:
        add_rect(slide, x, y, 14, 14, c, line=None)
    for x, y, c in [(880, 432, COLORS["gray_block"]), (894, 440, COLORS["blue"]), (908, 448, COLORS["gray_block"]), (894, 456, COLORS["gray_block"])]:
        add_rect(slide, x, y, 14, 14, c, line=None)


def add_page_no(slide, page_no: int) -> None:
    add_textbox(slide, 916, 502, 24, 18, f"{page_no:02d}", 10, COLORS["muted"], align=ALIGN_CENTER, font_name="Segoe UI")


def add_header(slide, tag: str, title: str, subtitle: str, page_no: int) -> None:
    add_bg(slide)
    add_logo_badge(slide)
    badge = add_round_box(slide, 34, 22, 138, 24, COLORS["white"], line=COLORS["blue"], line_weight=1.2)
    set_text(badge, tag, 9.5, COLORS["blue_dark"], bold=True, align=ALIGN_CENTER)
    add_textbox(slide, 42, 66, 560, 32, title, 25, COLORS["blue_dark"], bold=True)
    add_textbox(slide, 42, 98, 650, 24, subtitle, 11.2, COLORS["muted"])
    add_page_no(slide, page_no)


def add_card(slide, left: float, top: float, width: float, height: float, title: str, body_lines: list[str], *, badge: str | None = None, fill: tuple[int, int, int] = COLORS["white"], accent: tuple[int, int, int] = COLORS["blue"]):
    add_round_box(slide, left, top, width, height, fill, line=COLORS["line"], line_weight=1.1)
    add_rect(slide, left, top, width, 5, accent, line=None)
    title_x = left + 18
    if badge:
        add_circle(slide, left + 18, top + 14, 24, 24, accent, line=None)
        add_textbox(slide, left + 18, top + 16, 24, 20, badge, 9.5, COLORS["white"], bold=True, align=ALIGN_CENTER, font_name="Segoe UI")
        title_x = left + 50
    add_textbox(slide, title_x, top + 14, width - (title_x - left) - 12, 22, title, 15.5, COLORS["ink"], bold=True)
    body = "\n".join([f"• {line}" for line in body_lines])
    add_textbox(slide, left + 18, top + 46, width - 28, height - 54, body, 12.3, COLORS["muted"])


def add_step_box(slide, left: float, top: float, width: float, height: float, no: str, title: str, subtitle: str):
    add_round_box(slide, left, top, width, height, COLORS["white"], line=COLORS["line"], line_weight=1.1)
    add_rect(slide, left, top, width, 5, COLORS["blue"], line=None)
    add_textbox(slide, left + 18, top + 14, 24, 18, no, 10, COLORS["blue"], bold=True, font_name="Segoe UI")
    add_textbox(slide, left + 18, top + 32, width - 28, 24, title, 15, COLORS["blue_dark"], bold=True)
    add_textbox(slide, left + 18, top + 58, width - 28, height - 64, subtitle, 11, COLORS["muted"])


def add_chip(slide, left: float, top: float, width: float, height: float, text: str, *, fill: tuple[int, int, int] = COLORS["blue_lite"], text_color: tuple[int, int, int] = COLORS["blue_dark"]):
    chip = add_round_box(slide, left, top, width, height, fill, line=COLORS["line"], line_weight=0.9)
    set_text(chip, text, 11.5, text_color, bold=True, align=ALIGN_CENTER)
    return chip


def add_browser_frame(slide, left: float, top: float, width: float, height: float, image_path: Path | None):
    add_round_box(slide, left, top, width, height, COLORS["white"], line=COLORS["line"], line_weight=1.0)
    add_rect(slide, left, top, width, 24, COLORS["blue_dark"], line=None)
    add_circle(slide, left + 12, top + 7, 8, 8, COLORS["red_soft"], line=None)
    add_circle(slide, left + 26, top + 7, 8, 8, COLORS["gold_soft"], line=None)
    add_circle(slide, left + 40, top + 7, 8, 8, COLORS["green_soft"], line=None)
    add_image_or_placeholder(slide, image_path, left + 12, top + 32, width - 24, height - 44)


def add_metric_card(slide, left: float, top: float, width: float, height: float, label: str, value: str, note: str, *, accent: tuple[int, int, int], value_size: float = 28):
    add_round_box(slide, left, top, width, height, COLORS["white"], line=COLORS["line"])
    add_rect(slide, left, top, width, 5, accent, line=None)
    add_textbox(slide, left + 16, top + 16, width - 24, 18, label, 12, COLORS["muted"])
    add_textbox(slide, left + 16, top + 34, width - 24, 42, value, value_size, COLORS["blue_dark"], bold=True, font_name="Segoe UI Semibold")
    add_textbox(slide, left + 16, top + 76, width - 24, 20, note, 10.2, COLORS["muted"])


def add_bar_chart(slide, left: float, top: float, width: float, height: float):
    add_round_box(slide, left, top, width, height, COLORS["white"], line=COLORS["line"])
    add_textbox(slide, left + 18, top + 12, width - 36, 20, "真实样本统计", 13, COLORS["ink"], bold=True)
    axis_y = top + height - 34
    axis_x = left + 28
    axis_w = width - 56
    add_line(slide, axis_x, axis_y, axis_x + axis_w, axis_y, COLORS["line"], weight=1.2)
    values = [("人脸图片", 19, COLORS["blue"]), ("输入视频", 13, (242, 225, 210)), ("输出视频", 5, (224, 238, 223))]
    bar_w = 72
    gap = 52
    start_x = axis_x + 32
    max_v = 20
    for idx, (label, value, color) in enumerate(values):
        x = start_x + idx * (bar_w + gap)
        h = (height - 86) * value / max_v
        y = axis_y - h
        add_round_box(slide, x, y, bar_w, h, color, line=None)
        add_textbox(slide, x, y - 22, bar_w, 18, str(value), 12, COLORS["blue_dark"], bold=True, align=ALIGN_CENTER, font_name="Segoe UI")
        add_textbox(slide, x - 8, axis_y + 8, bar_w + 16, 22, label, 10.5, COLORS["muted"], align=ALIGN_CENTER)


def build_cover(slide):
    add_bg(slide)
    add_cover_decor(slide)
    add_logo_badge(slide, cover=True)
    add_textbox(slide, 330, 56, 300, 20, "软件工程专业毕业设计中期答辩", 11.5, COLORS["blue_dark"], bold=True, align=ALIGN_CENTER)
    add_line(slide, 60, 90, 208, 90, COLORS["blue"], weight=1.2)
    add_line(slide, 752, 90, 900, 90, COLORS["blue"], weight=1.2)
    add_textbox(slide, 252, 94, 460, 60, "人脸替换系统的", 27, COLORS["blue_dark"], bold=True, align=ALIGN_CENTER)
    add_textbox(slide, 302, 142, 360, 52, "设计与实现", 31, COLORS["blue_dark"], bold=True, align=ALIGN_CENTER)
    add_textbox(slide, 456, 188, 40, 16, "▼", 10, COLORS["blue"], align=ALIGN_CENTER, font_name="Segoe UI Symbol")
    add_textbox(slide, 86, 220, 248, 20, "答辩人：叶俊", 12, COLORS["blue_dark"])
    add_textbox(slide, 86, 246, 248, 20, "学院：软件学院", 12, COLORS["blue_dark"])
    add_textbox(slide, 86, 272, 248, 20, "学号：202200201151", 12, COLORS["blue_dark"])
    add_textbox(slide, 626, 220, 220, 20, "课题定位：系统实现型毕业设计", 12, COLORS["blue_dark"])
    add_textbox(slide, 626, 246, 220, 20, "阶段：中期答辩", 12, COLORS["blue_dark"])
    add_textbox(slide, 626, 272, 220, 20, "时间：2026 年 4 月", 12, COLORS["blue_dark"])
    add_round_box(slide, 94, 330, 772, 88, COLORS["white"], line=COLORS["line"])
    add_textbox(slide, 118, 348, 726, 44, "围绕 PyQt5 前端、Django REST 后端与人脸替换处理链路，展示系统实现、论文进展与答辩展示网站三项阶段成果。", 14, COLORS["muted"], align=ALIGN_CENTER)
    add_textbox(slide, 0, 512, 960, 14, "山东大学 · 软件学院", 10.5, COLORS["white"], align=ALIGN_CENTER)


def build_slide_2(slide):
    add_header(slide, "汇报提纲", "汇报内容", "按照“背景—思路—实现—结果—总结”的顺序依次汇报。", 2)
    items = [
        ("01", "研究背景", "说明选题意义与当前工作不足。"),
        ("02", "工作思路", "明确技术起点与本人完成内容。"),
        ("03", "系统实现", "展示架构、流程、界面与展示网站。"),
        ("04", "测试结果", "基于真实样本与真实记录进行说明。"),
        ("05", "阶段总结", "归纳成果、价值与下一步计划。"),
    ]
    positions = [(58, 156), (344, 156), (630, 156), (202, 316), (488, 316)]
    for (no, title, sub), (x, y) in zip(items, positions):
        add_step_box(slide, x, y, 230, 120, no, title, sub)
    add_round_box(slide, 146, 470, 668, 30, COLORS["blue_soft"], line=None)
    add_textbox(slide, 160, 476, 640, 14, "汇报时重点突出自己的工作，不在背景知识和他人方法上停留过久。", 10.5, COLORS["blue_dark"], bold=True, align=ALIGN_CENTER)


def build_slide_3(slide, temp_assets: dict[str, object]):
    add_header(slide, "中期进展", "当前进展与完成情况", "围绕系统完成度、论文完成内容与方法实现，说明中期阶段已完成的核心工作。", 3)

    add_card(
        slide,
        48,
        144,
        276,
        150,
        "系统完成内容",
        [
            "已完成视频模式、摄像头模式、结果回放与归档管理主流程。",
            "前端 PyQt5 + requests 与后端 Django REST 通信链路已跑通。",
            "支持可演示的线程化处理与状态反馈，不再是单次脚本。",
        ],
        badge="01",
        fill=COLORS["white"],
    )
    add_card(
        slide,
        342,
        144,
        276,
        150,
        "论文完成内容",
        [
            "已形成需求分析、总体设计、详细实现、测试分析主体章节。",
            "系统截图、样例图与答辩材料已按统一口径整理。",
            "论文内容与答辩讲稿主线一致，强调“系统实现与工程闭环”。",
        ],
        badge="02",
        fill=COLORS["blue_soft"],
    )
    add_card(
        slide,
        636,
        144,
        276,
        150,
        "方法实现说明",
        [
            "传统法：人脸检测/68点定位 → Delaunay 三角剖分。",
            "传统法：仿射变换拼接 → 无缝融合与颜色校正。",
            "AI法：InsightFace 检测对齐 → inswapper_128.onnx 推理。",
            "AI法：paste_back 回贴到原帧，并支持可选颜色平衡。",
        ],
        badge="03",
        fill=COLORS["green_soft"],
    )

    add_round_box(slide, 48, 312, 420, 160, COLORS["white"], line=COLORS["line"])
    add_rect(slide, 48, 312, 420, 4, COLORS["blue"], line=None)
    add_textbox(slide, 66, 318, 260, 18, "系统截图（运行界面）", 12.3, COLORS["ink"], bold=True)
    add_chip(slide, 342, 316, 110, 20, "系统实拍", fill=COLORS["blue_lite"])
    add_image_or_placeholder(
        slide,
        temp_assets.get("system_screenshot"),
        60,
        340,
        396,
        124,
        placeholder="系统截图缺失",
    )

    add_round_box(slide, 492, 312, 420, 160, COLORS["white"], line=COLORS["line"])
    add_rect(slide, 492, 312, 420, 4, COLORS["blue"], line=None)
    add_textbox(slide, 510, 318, 260, 18, "论文截图（章节插图）", 12.3, COLORS["ink"], bold=True)
    add_chip(slide, 750, 316, 146, 20, "论文第4章 图4.1", fill=COLORS["gold_soft"])
    add_image_or_placeholder(
        slide,
        temp_assets.get("thesis_screenshot"),
        504,
        340,
        396,
        124,
        placeholder="论文截图缺失",
    )

    add_round_box(slide, 80, 478, 800, 26, COLORS["gold_soft"], line=COLORS["line"])
    add_textbox(
        slide,
        98,
        482,
        764,
        18,
        "本页聚焦“做成了什么 + 论文写了什么 + 方法如何实现”，便于答辩现场直接讲系统完成度。",
        10.2,
        COLORS["blue_dark"],
        bold=True,
        align=ALIGN_CENTER,
    )


def build_slide_4(slide):
    add_header(slide, "问题切入", "现有工作的不足与课题切入点", "通过对比“常见不足”与“本课题切入点”，快速引出自己的工作价值。", 4)
    add_card(
        slide,
        56,
        148,
        328,
        286,
        "当前常见不足",
        [
            "偏脚本化：很多实现只能运行一次，不利于现场答辩展示。",
            "缺少素材管理：图片、视频和结果分散在本地目录，难以回查。",
            "缺少结果留痕：输出生成后难以统一回放、统计与归档。",
            "缺少交互反馈：处理过程、状态提示与异常信息不够直观。",
        ],
        badge="A",
        fill=COLORS["red_soft"],
        accent=(214, 98, 92),
    )
    add_textbox(slide, 420, 230, 120, 40, "因此", 22, COLORS["blue"], bold=True, align=ALIGN_CENTER)
    add_line(slide, 380, 250, 440, 250, COLORS["blue"], weight=2.0)
    add_line(slide, 520, 250, 580, 250, COLORS["blue"], weight=2.0)
    add_card(
        slide,
        576,
        148,
        328,
        286,
        "本课题切入点",
        [
            "构建 PyQt5 桌面前端，支持视频模式与摄像头模式两类工作流。",
            "引入 Django REST 与 SQLite，统一管理素材、任务与输出结果。",
            "打通“处理执行—结果保存—结果回放—数据归档”完整闭环。",
            "补充展示网站与论文材料，使答辩展示口径保持一致。",
        ],
        badge="B",
        fill=COLORS["green_soft"],
        accent=(110, 164, 92),
    )
    add_round_box(slide, 124, 454, 712, 30, COLORS["blue_soft"], line=None)
    add_textbox(slide, 140, 460, 680, 14, "课题价值不在单一算法指标，而在于把换脸能力组织成完整系统并形成可答辩的材料闭环。", 10.5, COLORS["blue_dark"], bold=True, align=ALIGN_CENTER)


def build_slide_5(slide):
    add_header(slide, "特色概览", "系统特色总览", "聚焦三项可答辩、可演示、可验证的系统特色能力。", 5)
    add_card(
        slide,
        48,
        154,
        272,
        170,
        "特色一：视频处理闭环",
        [
            "支持视频输入、参数校验、线程化处理与结果写出。",
            "处理完成后可直接回放并同步归档到数据库记录。",
            "形成“输入-处理-输出-验证”的完整业务链路。",
        ],
        badge="01",
        fill=COLORS["white"],
        accent=COLORS["blue"],
    )
    add_card(
        slide,
        344,
        154,
        272,
        170,
        "特色二：摄像头实时换脸",
        [
            "支持摄像头实时采集，并可按需启用实时换脸。",
            "支持现场切换目标人脸，即时观察效果变化。",
            "支持快照留痕，适配中期答辩实时演示场景。",
        ],
        badge="02",
        fill=COLORS["blue_soft"],
        accent=COLORS["blue_dark"],
    )
    add_card(
        slide,
        640,
        154,
        272,
        170,
        "特色三：结果留痕归档",
        [
            "图片、视频与输出结果统一记录在后端与数据库。",
            "支持查询、回放与统计，便于阶段汇报与复盘。",
            "证明项目是可持续迭代的系统而非一次性脚本。",
        ],
        badge="03",
        fill=COLORS["green_soft"],
        accent=(92, 151, 103),
    )
    add_card(
        slide,
        48,
        346,
        418,
        104,
        "支撑能力",
        [
            "双处理路径：InsightFace 主路径 + Traditional 兼容路径。",
            "界面与处理解耦：线程化任务保障界面响应。",
        ],
        fill=COLORS["white"],
    )
    add_card(
        slide,
        494,
        346,
        418,
        104,
        "答辩表达边界",
        [
            "本课题定位为系统实现型毕业设计，不宣称提出新换脸模型。",
            "汇报重点放在工程闭环、可演示性与阶段成果。",
        ],
        fill=COLORS["gold_soft"],
        accent=(212, 173, 89),
    )
    add_round_box(slide, 80, 468, 800, 30, COLORS["blue_soft"], line=COLORS["line"])
    add_textbox(slide, 96, 474, 768, 16, "一句话总结：系统既能离线处理视频，也能实时演示摄像头换脸，并具备结果留痕能力。", 10.2, COLORS["blue_dark"], bold=True, align=ALIGN_CENTER)


def build_slide_6(slide, temp_assets: dict[str, object]):
    add_header(slide, "技术路线", "系统架构图（前端 - 通信 - 后端）", "前后端分组展示技术栈，并明确系统通过 HTTP REST API 进行双向通信。", 6)
    icons = temp_assets["icons"]

    left_x = 48
    top_y = 144
    left_w = 270
    mid_w = 250
    right_w = 270
    gap = 37
    box_h = 302
    mid_x = left_x + left_w + gap
    right_x = mid_x + mid_w + gap

    add_round_box(slide, left_x, top_y, left_w, box_h, COLORS["white"], line=COLORS["line"])
    add_rect(slide, left_x, top_y, left_w, 5, COLORS["blue"], line=None)
    add_textbox(slide, left_x + 14, top_y + 12, left_w - 20, 20, "前端交互层", 14.2, COLORS["blue_dark"], bold=True)
    add_textbox(slide, left_x + 14, top_y + 34, left_w - 20, 16, "PyQt5 桌面端 + 本地处理线程", 9.8, COLORS["muted"])
    add_tech_item(slide, left_x + 14, top_y + 58, left_w - 28, 62, "PyQt5", "界面组织、模式切换、状态反馈", icons.get("qt"), "QT", fill=COLORS["white"])
    add_tech_item(slide, left_x + 14, top_y + 130, left_w - 28, 62, "OpenCV", "视频读取、帧处理与预览渲染", icons.get("opencv"), "CV", fill=COLORS["white"])
    add_tech_item(slide, left_x + 14, top_y + 202, left_w - 28, 62, "Python / Requests", "线程内发起 REST 接口调用", icons.get("python"), "Py", fill=COLORS["white"])

    add_round_box(slide, mid_x, top_y, mid_w, box_h, COLORS["blue_soft"], line=COLORS["line"])
    add_rect(slide, mid_x, top_y, mid_w, 5, COLORS["blue_dark"], line=None)
    add_textbox(slide, mid_x + 14, top_y + 12, mid_w - 20, 20, "通信与协议层", 14.2, COLORS["blue_dark"], bold=True)
    add_textbox(slide, mid_x + 14, top_y + 34, mid_w - 20, 16, "前后端共用本地 HTTP 通信", 9.8, COLORS["muted"])
    add_tech_item(slide, mid_x + 14, top_y + 58, mid_w - 28, 62, "HTTP REST API", "URL: http://localhost:8000/api", icons.get("openapi"), "API", fill=COLORS["white"], accent=COLORS["blue_dark"])
    add_tech_item(slide, mid_x + 14, top_y + 130, mid_w - 28, 62, "JSON", "图片/视频/任务元数据请求与响应", icons.get("json"), "JS", fill=COLORS["white"], accent=COLORS["blue_dark"])
    add_tech_item(slide, mid_x + 14, top_y + 202, mid_w - 28, 62, "multipart/form-data", "输出视频文件上传与回写", icons.get("openapi"), "UP", fill=COLORS["white"], accent=COLORS["blue_dark"])

    add_round_box(slide, right_x, top_y, right_w, box_h, COLORS["white"], line=COLORS["line"])
    add_rect(slide, right_x, top_y, right_w, 5, COLORS["blue"], line=None)
    add_textbox(slide, right_x + 14, top_y + 12, right_w - 20, 20, "后端处理与存储层", 14.2, COLORS["blue_dark"], bold=True)
    add_textbox(slide, right_x + 14, top_y + 34, right_w - 20, 16, "Django + SQLite + 媒体处理能力", 9.8, COLORS["muted"])
    add_tech_item(slide, right_x + 14, top_y + 58, right_w - 28, 62, "Django / DRF", "接口路由、资源管理、数据回写", icons.get("django"), "DJ", fill=COLORS["white"])
    add_tech_item(slide, right_x + 14, top_y + 130, right_w - 28, 62, "SQLite", "素材、任务、输出记录持久化", icons.get("sqlite"), "DB", fill=COLORS["white"])
    add_tech_item(slide, right_x + 14, top_y + 202, right_w - 28, 62, "ONNX + OpenCV", "模型推理与视频编解码处理", icons.get("onnx"), "NX", fill=COLORS["white"])

    add_arrow_line(slide, left_x + left_w + 3, top_y + 170, mid_x - 3, top_y + 170, COLORS["blue_dark"], weight=1.8, bidirectional=True)
    add_textbox(slide, left_x + left_w + 8, top_y + 152, gap - 14, 14, "请求/响应", 8.6, COLORS["blue_dark"], align=ALIGN_CENTER, font_name="Segoe UI")
    add_arrow_line(slide, mid_x + mid_w + 3, top_y + 170, right_x - 3, top_y + 170, COLORS["blue_dark"], weight=1.8, bidirectional=True)
    add_textbox(slide, mid_x + mid_w + 8, top_y + 152, gap - 14, 14, "元数据/文件", 8.3, COLORS["blue_dark"], align=ALIGN_CENTER, font_name="Segoe UI")

    add_round_box(slide, 76, 468, 808, 30, COLORS["gold_soft"], line=COLORS["line"])
    add_textbox(slide, 96, 474, 768, 16, "视频模式与摄像头模式共用同一通信链路：前端 requests ⇄ HTTP REST API ⇄ Django/SQLite。", 10.2, COLORS["blue_dark"], bold=True, align=ALIGN_CENTER)


def build_slide_7(slide):
    add_header(slide, "个人工作", "我的主要实现工作", "详细工作部分突出思路和重点，明确哪些内容是本人完成的工程实现。", 7)
    add_card(
        slide,
        58,
        148,
        400,
        132,
        "01 前端界面组织",
        [
            "整理主界面区域布局，支持视频模式与摄像头模式切换。",
            "将素材列表、参数区、预览区和状态反馈整合到单窗口中。",
        ],
        fill=COLORS["white"],
    )
    add_card(
        slide,
        500,
        148,
        400,
        132,
        "02 处理流程与线程控制",
        [
            "打通 VideoProcessingThread 与 CameraProcessingThread 两类流程。",
            "保证处理时界面仍可响应，并反馈进度、状态和结果。",
        ],
        fill=COLORS["blue_soft"],
    )
    add_card(
        slide,
        58,
        304,
        400,
        132,
        "03 资源管理与结果回放",
        [
            "通过 DatabaseManager 与后端接口管理图片、视频与输出记录。",
            "实现结果保存、回放验证和数据库归档的一致流程。",
        ],
        fill=COLORS["green_soft"],
    )
    add_card(
        slide,
        500,
        304,
        400,
        132,
        "04 展示网站与论文材料整理",
        [
            "制作答辩展示网站，统一呈现背景、功能、架构、测试和展望。",
            "将系统截图、真实数据与样例图整理进论文和答辩材料。",
        ],
        fill=COLORS["gold_soft"],
    )
    add_round_box(slide, 86, 462, 786, 30, COLORS["blue_soft"], line=None)
    add_textbox(slide, 106, 468, 746, 16, "这部分建议口头强调“我做了哪些系统化工作”，不要把重心放在他人算法原理上。", 10.2, COLORS["blue_dark"], bold=True, align=ALIGN_CENTER)


def build_slide_8(slide):
    add_header(slide, "流程一", "视频模式处理闭环（进度重点）", "中期阶段已跑通“输入素材→线程处理→输出回写→结果回放”的完整流程。", 8)
    add_round_box(slide, 48, 144, 864, 118, COLORS["white"], line=COLORS["line"])
    add_picture_fit(slide, ASSETS["flow_video"], 62, 168, 836, 72)
    add_card(
        slide,
        48,
        292,
        268,
        136,
        "输入准备",
        [
            "选择目标人脸与待处理视频，完成本地可用性校验。",
            "设置处理方法、输出路径与基础参数。",
        ],
        badge="01",
        fill=COLORS["white"],
    )
    add_card(
        slide,
        346,
        292,
        268,
        136,
        "处理执行",
        [
            "启动 VideoProcessingThread，界面保持可响应。",
            "后台持续处理帧并写出输出视频。",
        ],
        badge="02",
        fill=COLORS["blue_soft"],
    )
    add_card(
        slide,
        644,
        292,
        268,
        136,
        "结果回写",
        [
            "处理结束后记录输出文件并触发数据库归档。",
            "支持立即回放，完成结果可验证闭环。",
        ],
        badge="03",
        fill=COLORS["green_soft"],
    )
    add_textbox(slide, 48, 456, 864, 18, "讲解重点：强调“线程化处理 + 输出回写 + 回放验证”的工程闭环，而非单帧效果演示。", 10.5, COLORS["muted"], align=ALIGN_CENTER)


def build_slide_9(slide):
    add_header(slide, "流程二", "摄像头模式与实时演示（进度重点）", "中期阶段已支持摄像头实时采集、可选换脸和快照留痕，适配现场答辩演示。", 9)
    add_round_box(slide, 48, 144, 864, 118, COLORS["white"], line=COLORS["line"])
    add_picture_fit(slide, ASSETS["flow_camera"], 62, 168, 836, 72)
    add_card(
        slide,
        90,
        294,
        240,
        122,
        "实时采集",
        [
            "启动摄像头前自动停止视频播放，避免资源冲突。",
            "持续获取实时帧并更新预览区。",
        ],
        badge="01",
    )
    add_card(
        slide,
        360,
        294,
        240,
        122,
        "可选换脸",
        [
            "可根据演示需要决定是否开启实时换脸。",
            "更换目标人脸后可即时观察结果变化。",
        ],
        badge="02",
        fill=COLORS["blue_soft"],
    )
    add_card(
        slide,
        630,
        294,
        240,
        122,
        "快照与留痕",
        [
            "支持快照保存，形成可复用演示素材。",
            "便于现场展示后补充到论文或答辩材料中。",
        ],
        badge="03",
        fill=COLORS["gold_soft"],
    )
    add_round_box(slide, 136, 452, 688, 30, COLORS["blue_soft"], line=None)
    add_textbox(slide, 150, 458, 660, 16, "与视频模式相比，摄像头模式更强调低延迟响应、状态可见性与现场可讲解性。", 10.2, COLORS["blue_dark"], bold=True, align=ALIGN_CENTER)


def build_slide_10(slide):
    add_header(slide, "界面展示", "系统界面与处理效果", "真实界面截图和样例图比大量文字更能说明系统是否真正做出来了。", 10)
    add_round_box(slide, 48, 144, 520, 304, COLORS["white"], line=COLORS["line"])
    add_textbox(slide, 66, 160, 200, 18, "系统主界面", 13, COLORS["ink"], bold=True)
    add_picture_fit(slide, ASSETS["ui_main"], 62, 184, 492, 246)
    add_round_box(slide, 598, 144, 314, 166, COLORS["white"], line=COLORS["line"])
    add_textbox(slide, 616, 160, 260, 18, "样例：输入图像与换脸结果", 13, COLORS["ink"], bold=True)
    add_picture_fit(slide, ASSETS["pair_a"], 612, 188, 286, 104)
    add_card(
        slide,
        598,
        332,
        314,
        116,
        "展示说明",
        [
            "界面集成了素材选择、参数设置、模式切换与状态反馈。",
            "样例图直接展示系统已经具备可视化处理结果。",
        ],
        fill=COLORS["blue_soft"],
    )
    add_round_box(slide, 74, 468, 812, 28, COLORS["gold_soft"], line=None)
    add_textbox(slide, 90, 474, 780, 14, "如现场允许，可补充演示视频模式或摄像头模式的实际运行过程。", 10.2, COLORS["blue_dark"], bold=True, align=ALIGN_CENTER)


def build_slide_11(slide, temp_assets: dict[str, object]):
    add_header(slide, "网站成果", "展示网站成果", "答辩展示网站用于统一整理系统背景、功能、测试分析和总结展望。", 11)
    add_browser_frame(slide, 48, 144, 360, 316, temp_assets["website_top"])
    add_textbox(slide, 48, 470, 360, 18, "网站首页真实渲染截图", 10.5, COLORS["muted"], align=ALIGN_CENTER)
    add_card(
        slide,
        442,
        144,
        470,
        120,
        "网站承担的作用",
        [
            "集中呈现系统背景、功能、技术路线、界面展示、测试分析和总结展望。",
            "帮助评委在浏览答辩材料时快速获取完整信息，也便于现场顺序讲解。",
        ],
        fill=COLORS["white"],
    )
    add_textbox(slide, 442, 290, 180, 18, "页面主要章节", 13, COLORS["ink"], bold=True)
    chips = [
        ("系统功能", 442, 318), ("技术路线", 588, 318), ("界面展示", 734, 318),
        ("测试分析", 516, 364), ("总结展望", 662, 364),
    ]
    for text, x, y in chips:
        add_chip(slide, x, y, 126, 34, text)
    add_round_box(slide, 442, 418, 470, 42, COLORS["blue_soft"], line=COLORS["line"])
    add_textbox(slide, 462, 428, 430, 16, "网站内容与系统、论文使用同一套真实材料，不使用模板示例文字或无关下载标识。", 10.2, COLORS["blue_dark"], bold=True, align=ALIGN_CENTER)


def build_slide_12(slide):
    add_header(slide, "论文进度", "论文完成情况与内容一致性", "论文内容需要与系统和答辩口径保持一致，这是答辩环节非常重要的一点。", 12)
    chapters = [
        ("01", "摘要与关键词"), ("02", "绪论"),
        ("03", "相关技术"), ("04", "需求分析"),
        ("05", "总体设计"), ("06", "详细实现"),
        ("07", "测试分析"), ("08", "总结展望"),
    ]
    positions = [(48, 150), (286, 150), (48, 232), (286, 232), (48, 314), (286, 314), (48, 396), (286, 396)]
    fills = [COLORS["blue_lite"], COLORS["blue_soft"], COLORS["green_soft"], COLORS["gold_soft"], (248, 235, 220), (240, 235, 250), COLORS["blue_lite"], COLORS["blue_soft"]]
    for (no, title), (x, y), fill in zip(chapters, positions, fills):
        add_round_box(slide, x, y, 206, 64, fill, line=COLORS["line"])
        add_textbox(slide, x + 18, y + 12, 40, 18, no, 10, COLORS["muted"], bold=True, font_name="Segoe UI")
        add_textbox(slide, x + 18, y + 28, 164, 22, title, 15, COLORS["blue_dark"], bold=True)
    add_card(
        slide,
        560,
        150,
        352,
        164,
        "当前完成状态",
        [
            "论文题目、中英文摘要与目录结构已经完整。",
            "需求分析、总体设计、详细实现与测试分析均已形成正文。",
            "系统截图、真实样例图和真实统计数据已纳入论文材料。",
        ],
        fill=COLORS["white"],
    )
    add_card(
        slide,
        560,
        336,
        352,
        110,
        "口径一致性",
        [
            "论文强调系统实现与工程验证，不夸大算法创新。",
            "答辩页面中的数据与论文正文保持一致。",
        ],
        fill=COLORS["blue_soft"],
    )
    add_round_box(slide, 560, 462, 352, 30, COLORS["gold_soft"], line=None)
    add_textbox(slide, 576, 468, 320, 16, "论文写了的内容，答辩 PPT 中都应有所体现。", 10.2, COLORS["blue_dark"], bold=True, align=ALIGN_CENTER)


def build_slide_13(slide, temp_assets: dict[str, object]):
    add_header(slide, "中期结论", "中期结论与效果样例对比", "不罗列统计数字，直接展示传统方法与 AI 方法在真实样例中的换脸效果。", 13)

    traditional_input = temp_assets.get("traditional_input_frame")
    traditional_output = temp_assets.get("traditional_output_frame")
    ai_output = temp_assets.get("ai_sample_frame")
    ai_note = str(temp_assets.get("ai_sample_note", ""))

    add_round_box(slide, 48, 144, 414, 320, COLORS["white"], line=COLORS["line"])
    add_rect(slide, 48, 144, 414, 5, COLORS["blue"], line=None)
    add_textbox(slide, 66, 156, 378, 22, "传统三角剖分方法（真实样例）", 15, COLORS["ink"], bold=True)

    add_round_box(slide, 64, 188, 182, 228, COLORS["paper"], line=COLORS["line"])
    add_image_or_placeholder(slide, traditional_input, 74, 198, 162, 176, placeholder="输入帧缺失")
    add_textbox(slide, 74, 384, 162, 24, "输入帧", 11, COLORS["muted"], align=ALIGN_CENTER)

    add_round_box(slide, 262, 188, 182, 228, COLORS["paper"], line=COLORS["line"])
    add_image_or_placeholder(slide, traditional_output, 272, 198, 162, 176, placeholder="传统输出缺失")
    add_textbox(slide, 272, 384, 162, 24, "传统法输出", 11, COLORS["muted"], align=ALIGN_CENTER)

    add_round_box(slide, 498, 144, 414, 320, COLORS["white"], line=COLORS["line"])
    add_rect(slide, 498, 144, 414, 5, (92, 151, 103), line=None)
    add_textbox(slide, 516, 156, 378, 22, "AI 模型方法（真实样例）", 15, COLORS["ink"], bold=True)

    add_round_box(slide, 514, 188, 182, 228, COLORS["paper"], line=COLORS["line"])
    add_image_or_placeholder(slide, traditional_input, 524, 198, 162, 176, placeholder="输入帧缺失")
    add_textbox(slide, 524, 384, 162, 24, "输入帧", 11, COLORS["muted"], align=ALIGN_CENTER)

    add_round_box(slide, 712, 188, 182, 228, COLORS["paper"], line=COLORS["line"])
    add_image_or_placeholder(slide, ai_output, 722, 198, 162, 176, placeholder="AI 输出缺失")
    add_textbox(slide, 722, 384, 162, 24, "AI 法输出", 11, COLORS["muted"], align=ALIGN_CENTER)

    add_textbox(slide, 508, 426, 394, 20, ai_note, 9.4, COLORS["muted"], align=ALIGN_CENTER)

    add_round_box(slide, 66, 474, 828, 28, COLORS["blue_soft"], line=COLORS["line"])
    add_textbox(
        slide,
        84,
        478,
        792,
        18,
        "结论：传统方法可解释性更强，AI 方法在真实场景下融合更自然，适合作为答辩主演示链路。",
        10.1,
        COLORS["blue_dark"],
        bold=True,
        align=ALIGN_CENTER,
    )


def build_slide_14(slide):
    add_header(slide, "后续安排", "下一步计划与功能完善", "按阶段推进功能完善与答辩交付，重点补齐上传、删除、回放和稳定性能力。", 14)
    add_card(
        slide,
        48,
        148,
        272,
        280,
        "近期计划（1-2 周）",
        [
            "增加上传视频删除功能：前端提供删除入口，后端补充删除接口。",
            "删除时同步处理数据库记录与本地文件，避免脏数据残留。",
            "完善输出记录删除/清理功能，支持按条删除与批量清理。",
            "优化上传校验与列表刷新，减少重复上传与状态错乱。",
        ],
        badge="01",
        fill=COLORS["white"],
    )
    add_card(
        slide,
        344,
        148,
        272,
        280,
        "中期计划（答辩前）",
        [
            "补强视频任务状态流转与异常恢复机制，提升长任务稳定性。",
            "完善进度提示与失败原因反馈，降低现场演示不确定性。",
            "优化摄像头模式的交互细节（切换提示、快照管理、状态可视化）。",
            "补充更多真实样例，固定传统法与 AI 法的对比展示素材。",
        ],
        badge="02",
        fill=COLORS["blue_soft"],
    )
    add_card(
        slide,
        640,
        148,
        272,
        280,
        "阶段收口（答辩周）",
        [
            "固化演示链路：视频模式与摄像头模式双场景一键切换预案。",
            "统一讲稿口径：系统完成度、方法实现、样例对比三条主线。",
            "确保 PPT、论文与展示网站描述一致，减少答辩现场口径偏差。",
            "完成答辩版回归检查清单，保证“可演示、可讲解、可追溯”。",
        ],
        badge="03",
        fill=COLORS["gold_soft"],
    )

    add_line(slide, 322, 286, 344, 286, COLORS["line"], weight=1.4)
    add_line(slide, 616, 286, 640, 286, COLORS["line"], weight=1.4)
    add_textbox(slide, 326, 276, 14, 16, "→", 11, COLORS["muted"], align=ALIGN_CENTER, font_name="Segoe UI")
    add_textbox(slide, 620, 276, 14, 16, "→", 11, COLORS["muted"], align=ALIGN_CENTER, font_name="Segoe UI")

    add_round_box(slide, 48, 440, 864, 48, COLORS["green_soft"], line=COLORS["line"])
    add_textbox(
        slide,
        66,
        450,
        828,
        30,
        "本页仅给出可执行的下一步计划，不做风险罗列；重点是把“可演示系统”继续完善为“可交付系统”。",
        10.6,
        COLORS["blue_dark"],
        bold=True,
        align=ALIGN_CENTER,
    )

    add_round_box(slide, 104, 498, 752, 22, COLORS["blue"], line=None)
    add_textbox(slide, 124, 502, 712, 14, "阶段目标：围绕功能完整性与演示稳定性推进，确保答辩讲解顺畅。", 9.8, COLORS["white"], bold=True, align=ALIGN_CENTER)


def build_slide_15(slide):
    add_bg(slide)
    add_cover_decor(slide)
    add_logo_badge(slide, cover=True)
    add_textbox(slide, 0, 180, 960, 44, "谢谢各位老师", 28, COLORS["blue_dark"], bold=True, align=ALIGN_CENTER)
    add_textbox(slide, 0, 234, 960, 24, "敬请批评指正", 14, COLORS["muted"], align=ALIGN_CENTER)
    add_round_box(slide, 238, 312, 484, 38, COLORS["white"], line=COLORS["line"])
    add_textbox(slide, 258, 322, 444, 16, "最后，感谢所有关心和帮助过我的老师与同学。", 11, COLORS["blue_dark"], bold=True, align=ALIGN_CENTER)


def build_presentation():
    temp_assets = prepare_temp_assets()
    out_pptx = OUT_PPTX
    out_pdf = OUT_PDF
    if not safe_unlink(out_pptx):
        out_pptx = ROOT / f"{OUT_PPTX.stem}_regenerated.pptx"
        safe_unlink(out_pptx)
    if not safe_unlink(out_pdf):
        out_pdf = ROOT / f"{OUT_PDF.stem}_regenerated.pdf"
        safe_unlink(out_pdf)
    shutil.rmtree(PREVIEW_DIR, ignore_errors=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    app = win32com.client.Dispatch("PowerPoint.Application")
    app.Visible = True
    app.DisplayAlerts = 0
    presentation = app.Presentations.Add()
    try:
        presentation.PageSetup.SlideWidth = SLIDE_W
        presentation.PageSetup.SlideHeight = SLIDE_H
        slides = [presentation.Slides.Add(i + 1, PP_LAYOUT_BLANK) for i in range(15)]
        build_cover(slides[0])
        build_slide_2(slides[1])
        build_slide_3(slides[2], temp_assets)
        build_slide_4(slides[3])
        build_slide_5(slides[4])
        build_slide_6(slides[5], temp_assets)
        build_slide_7(slides[6])
        build_slide_8(slides[7])
        build_slide_9(slides[8])
        build_slide_10(slides[9])
        build_slide_11(slides[10], temp_assets)
        build_slide_12(slides[11])
        build_slide_13(slides[12], temp_assets)
        build_slide_14(slides[13])
        build_slide_15(slides[14])
        presentation.SaveAs(str(out_pptx), PPTX_FORMAT)
        presentation.SaveAs(str(out_pdf), PDF_FORMAT)
        presentation.Export(str(PREVIEW_DIR), "PNG", 1600, 900)
        print(f"PPTX: {out_pptx}")
        print(f"PDF: {out_pdf}")
        print(f"PreviewDir: {PREVIEW_DIR}")
    finally:
        presentation.Close()
        app.Quit()


if __name__ == "__main__":
    build_presentation()
