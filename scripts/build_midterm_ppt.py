# -*- coding: utf-8 -*-
from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image
import win32com.client


ROOT = Path(__file__).resolve().parents[1]
OUT_PPTX = ROOT / "midterm_face_swap_defense.pptx"
OUT_PDF = ROOT / "midterm_face_swap_defense.pdf"
PREVIEW_DIR = ROOT / "debug" / "midterm_face_swap_defense_preview"
TEMP_DIR = ROOT / "thesis_assets_generated" / "ppt_temp"

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
    "paper": (244, 248, 252),
    "white": (255, 255, 255),
    "navy": (12, 27, 48),
    "navy_soft": (20, 40, 68),
    "navy_card": (23, 44, 74),
    "ink": (22, 39, 60),
    "muted": (89, 107, 129),
    "line": (202, 217, 232),
    "cyan": (24, 195, 217),
    "cyan_soft": (232, 250, 252),
    "cyan_mid": (128, 223, 234),
    "blue_soft": (229, 239, 250),
    "green_soft": (234, 245, 232),
    "yellow_soft": (255, 244, 214),
    "peach_soft": (252, 237, 226),
    "red_soft": (255, 233, 234),
    "purple_soft": (241, 236, 251),
    "white_90": (245, 251, 255),
}


ASSETS = {
    "ui_main": ROOT / "thesis_assets_generated" / "ui_main.png",
    "pair_a": ROOT / "thesis_assets_generated" / "pair_57272a8d42.png",
    "architecture": ROOT / "thesis_assets_generated" / "two_row_9f04ed605a.png",
    "website_full": ROOT / "thesis_assets_generated" / "website_home.png",
}


def rgb(color: tuple[int, int, int]) -> int:
    r, g, b = color
    return r + (g << 8) + (b << 16)


def safe_unlink(path: Path) -> None:
    if path.exists():
        path.unlink()


def prepare_temp_assets() -> dict[str, Path]:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    website_crop = TEMP_DIR / "website_top.png"
    with Image.open(ASSETS["website_full"]) as img:
        cropped = img.crop((0, 0, 1600, 2200))
        cropped.save(website_crop)

    return {
        "website_top": website_crop,
    }


def fit_box(img_w: int, img_h: int, box_w: float, box_h: float) -> tuple[float, float]:
    scale = min(box_w / img_w, box_h / img_h)
    return img_w * scale, img_h * scale


def set_text(shape, text: str, font_size: float, color: tuple[int, int, int], *, bold: bool = False, align: int = ALIGN_LEFT, font_name: str = "Microsoft YaHei") -> None:
    tf = shape.TextFrame
    tf.TextRange.Text = text
    tf.MarginLeft = 6
    tf.MarginRight = 6
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


def add_shape(slide, shape_type: int, left: float, top: float, width: float, height: float, fill: tuple[int, int, int], *, line: tuple[int, int, int] | None = None, transparency: float = 0.0):
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
        shape.Line.Weight = 1.25
    return shape


def add_round_box(slide, left: float, top: float, width: float, height: float, fill: tuple[int, int, int], *, line: tuple[int, int, int] | None = None, transparency: float = 0.0):
    return add_shape(slide, SHAPE_ROUND, left, top, width, height, fill, line=line, transparency=transparency)


def add_rect(slide, left: float, top: float, width: float, height: float, fill: tuple[int, int, int], *, line: tuple[int, int, int] | None = None, transparency: float = 0.0):
    return add_shape(slide, SHAPE_RECT, left, top, width, height, fill, line=line, transparency=transparency)


def add_circle(slide, left: float, top: float, width: float, height: float, fill: tuple[int, int, int], *, line: tuple[int, int, int] | None = None, transparency: float = 0.0):
    return add_shape(slide, SHAPE_OVAL, left, top, width, height, fill, line=line, transparency=transparency)


def add_picture_fit(slide, path: Path, left: float, top: float, width: float, height: float):
    with Image.open(path) as img:
        img_w, img_h = img.size
    new_w, new_h = fit_box(img_w, img_h, width, height)
    pic_left = left + (width - new_w) / 2
    pic_top = top + (height - new_h) / 2
    return slide.Shapes.AddPicture(str(path), False, True, pic_left, pic_top, new_w, new_h)


def add_slide_bg(slide, *, dark: bool = False) -> None:
    slide.FollowMasterBackground = False
    fill = slide.Background.Fill
    fill.Visible = -1
    fill.Solid()
    fill.ForeColor.RGB = rgb(COLORS["navy"] if dark else COLORS["paper"])

    if dark:
        add_circle(slide, 690, -70, 210, 210, COLORS["cyan"], transparency=0.84)
        add_circle(slide, 760, 360, 240, 240, COLORS["cyan_mid"], transparency=0.9)
        add_circle(slide, -70, 400, 160, 160, COLORS["white"], transparency=0.92)
        return

    add_rect(slide, 0, 0, SLIDE_W, 8, COLORS["navy"])
    add_circle(slide, 760, -120, 220, 220, COLORS["cyan"], transparency=0.9)
    add_circle(slide, -60, 430, 170, 170, COLORS["cyan_mid"], transparency=0.92)


def add_page_no(slide, num: int) -> None:
    add_textbox(slide, 900, 500, 36, 20, f"{num:02d}", 10, COLORS["muted"], align=ALIGN_CENTER, font_name="Segoe UI")


def add_title_block(slide, eyebrow: str, title: str, subtitle: str, page_no: int) -> None:
    add_tag(slide, 48, 28, 136, 22, eyebrow, COLORS["cyan_soft"], COLORS["cyan"], COLORS["navy"], font_size=9.5)
    add_textbox(slide, 48, 58, 700, 34, title, 26, COLORS["ink"], bold=True)
    add_textbox(slide, 48, 90, 760, 28, subtitle, 11.5, COLORS["muted"])
    add_page_no(slide, page_no)


def add_tag(slide, left: float, top: float, width: float, height: float, text: str, fill: tuple[int, int, int], line: tuple[int, int, int], text_color: tuple[int, int, int], *, font_size: float = 10.0):
    tag = add_round_box(slide, left, top, width, height, fill, line=line)
    set_text(tag, text, font_size, text_color, bold=True, align=ALIGN_CENTER)
    return tag


def add_card(slide, left: float, top: float, width: float, height: float, title: str, body_lines: list[str], accent: tuple[int, int, int], *, fill: tuple[int, int, int] = COLORS["white"], badge: str | None = None, title_color: tuple[int, int, int] = COLORS["ink"], body_color: tuple[int, int, int] = COLORS["muted"]):
    add_round_box(slide, left, top, width, height, fill, line=COLORS["line"])
    add_rect(slide, left, top, width, 6, accent)
    title_left = left + 20
    if badge:
        add_circle(slide, left + 18, top + 16, 26, 26, accent)
        add_textbox(slide, left + 18, top + 18, 26, 22, badge, 10.5, COLORS["white"], bold=True, align=ALIGN_CENTER, font_name="Segoe UI")
        title_left = left + 56
    add_textbox(slide, title_left, top + 16, width - (title_left - left) - 16, 28, title, 17, title_color, bold=True)
    body = "\n".join([f"• {line}" for line in body_lines])
    add_textbox(slide, left + 20, top + 52, width - 32, height - 62, body, 12.5, body_color)


def add_metric_card(slide, left: float, top: float, width: float, height: float, label: str, value: str, accent: tuple[int, int, int], note: str, *, value_size: float = 28, note_size: float = 10.5):
    add_round_box(slide, left, top, width, height, COLORS["white"], line=COLORS["line"])
    add_rect(slide, left, top, width, 6, accent)
    add_textbox(slide, left + 18, top + 16, width - 36, 20, label, 12, COLORS["muted"])
    add_textbox(slide, left + 18, top + 34, width - 36, 42, value, value_size, COLORS["ink"], bold=True, font_name="Segoe UI Semibold")
    add_textbox(slide, left + 18, top + 76, width - 36, 22, note, note_size, COLORS["muted"])


def add_chip(slide, left: float, top: float, width: float, height: float, text: str, fill: tuple[int, int, int]):
    chip = add_round_box(slide, left, top, width, height, fill, line=COLORS["line"])
    set_text(chip, text, 12, COLORS["ink"], bold=True, align=ALIGN_CENTER)
    return chip


def add_browser_frame(slide, left: float, top: float, width: float, height: float, image_path: Path):
    add_round_box(slide, left, top, width, height, COLORS["white"], line=COLORS["line"])
    add_rect(slide, left, top, width, 26, COLORS["navy"])
    add_circle(slide, left + 14, top + 7, 8, 8, COLORS["red_soft"])
    add_circle(slide, left + 28, top + 7, 8, 8, COLORS["yellow_soft"])
    add_circle(slide, left + 42, top + 7, 8, 8, COLORS["green_soft"])
    add_picture_fit(slide, image_path, left + 12, top + 36, width - 24, height - 48)


def draw_architecture_flow(slide, left: float, top: float, width: float):
    box_w = 188
    box_h = 86
    gap = 18
    x1 = left
    x2 = left + box_w + gap
    x3 = left
    x4 = left + box_w + gap
    y1 = top
    y2 = top
    y3 = top + box_h + 40
    y4 = y3

    add_card(slide, x1, y1, box_w, box_h, "PyQt5 前端界面", ["视频模式 / 摄像头模式", "素材选择与状态反馈"], COLORS["cyan"])
    add_card(slide, x2, y2, box_w, box_h, "业务控制层", ["模式切换与参数同步", "线程启动与结果回写"], COLORS["blue_soft"])
    add_card(slide, x3, y3, box_w, box_h, "处理引擎", ["OpenCV 帧处理", "InsightFace / Traditional"], COLORS["peach_soft"])
    add_card(slide, x4, y4, box_w, box_h, "数据管理层", ["Django REST + SQLite", "本地媒体目录与归档"], COLORS["green_soft"])

    line1 = slide.Shapes.AddLine(x1 + box_w, y1 + box_h / 2, x2, y2 + box_h / 2)
    line2 = slide.Shapes.AddLine(x2 + box_w / 2, y2 + box_h, x4 + box_w / 2, y4)
    line3 = slide.Shapes.AddLine(x1 + box_w / 2, y1 + box_h, x3 + box_w / 2, y3)
    line4 = slide.Shapes.AddLine(x3 + box_w, y3 + box_h / 2, x4, y4 + box_h / 2)
    for line in (line1, line2, line3, line4):
        line.Line.ForeColor.RGB = rgb(COLORS["cyan"])
        line.Line.Weight = 2.0

    add_textbox(slide, left + 24, top + 196, width - 48, 22, "前端交互层 + 业务控制层 + 媒体处理层 + 数据持久化层", 11, COLORS["muted"], align=ALIGN_CENTER)


def add_bar_chart(slide, left: float, top: float, width: float, height: float):
    add_round_box(slide, left, top, width, height, COLORS["white"], line=COLORS["line"])
    add_textbox(slide, left + 18, top + 14, width - 36, 20, "真实样本记录数量", 13, COLORS["ink"], bold=True)

    chart_left = left + 24
    chart_top = top + 50
    chart_w = width - 48
    chart_h = height - 82
    axis = slide.Shapes.AddLine(chart_left, chart_top + chart_h, chart_left + chart_w, chart_top + chart_h)
    axis.Line.ForeColor.RGB = rgb(COLORS["line"])
    axis.Line.Weight = 1.5

    labels = [("人脸图片", 19, COLORS["cyan"]), ("输入视频", 13, COLORS["peach_soft"]), ("输出视频", 5, COLORS["green_soft"])]
    max_value = 20
    bar_w = 68
    gap = 44
    start_x = chart_left + 22
    for idx, (label, value, color) in enumerate(labels):
        bar_left = start_x + idx * (bar_w + gap)
        bar_h = (chart_h - 30) * value / max_value
        bar_top = chart_top + chart_h - bar_h
        add_round_box(slide, bar_left, bar_top, bar_w, bar_h, color, line=None)
        add_textbox(slide, bar_left, bar_top - 22, bar_w, 20, str(value), 12, COLORS["ink"], bold=True, align=ALIGN_CENTER, font_name="Segoe UI Semibold")
        add_textbox(slide, bar_left - 12, chart_top + chart_h + 6, bar_w + 24, 24, label, 10.5, COLORS["muted"], align=ALIGN_CENTER)


def build_cover(slide):
    add_slide_bg(slide)
    add_circle(slide, 650, 24, 260, 260, COLORS["cyan"], transparency=0.92)
    add_round_box(slide, 42, 58, 538, 356, COLORS["navy"], line=None)
    add_tag(slide, 72, 86, 210, 24, "软件工程专业毕业设计中期答辩", COLORS["cyan_soft"], COLORS["cyan"], COLORS["navy"], font_size=9)
    add_textbox(slide, 72, 126, 420, 82, "人脸替换系统的\n设计与实现", 28, COLORS["white_90"], bold=True)
    add_textbox(slide, 72, 214, 450, 44, "围绕已实现的 PyQt5 前端、Django REST 后端与换脸处理链路，汇报系统、论文与展示网站的阶段性成果。", 13, COLORS["white"], font_name="Microsoft YaHei")

    add_card(slide, 72, 286, 144, 76, "系统成果", ["双模式交互", "结果回放与归档"], COLORS["cyan"], fill=COLORS["navy_card"], title_color=COLORS["white"], body_color=COLORS["white"])
    add_card(slide, 228, 286, 144, 76, "论文成果", ["完整章节结构", "聚焦系统实现"], COLORS["cyan_mid"], fill=COLORS["navy_card"], title_color=COLORS["white"], body_color=COLORS["white"])
    add_card(slide, 384, 286, 144, 76, "网站成果", ["展示页已完成", "辅助现场讲解"], COLORS["cyan_soft"], fill=COLORS["navy_card"], title_color=COLORS["white"], body_color=COLORS["white"])

    add_round_box(slide, 632, 78, 276, 254, COLORS["white"], line=COLORS["line"])
    add_textbox(slide, 656, 106, 220, 24, "答辩信息", 15, COLORS["ink"], bold=True)
    add_rect(slide, 656, 138, 196, 1.5, COLORS["line"])
    add_textbox(slide, 656, 154, 220, 120, "题目：人脸替换系统的设计与实现\n姓名：叶俊\n学号：202200201151\n学院：软件学院\n日期：2026年4月", 13, COLORS["muted"])
    add_textbox(slide, 72, 392, 390, 24, "系统实现型毕业设计，重点展示工程闭环与阶段成果。", 11, COLORS["white"])


def build_slide_2(slide):
    add_slide_bg(slide)
    add_title_block(slide, "课题定位", "课题背景与研究目标", "从算法原型走向可展示、可管理、可归档的完整系统。", 2)

    add_card(
        slide,
        48,
        126,
        354,
        196,
        "研究背景",
        [
            "许多人脸替换项目停留在单次脚本或效果演示层面。",
            "素材导入、状态反馈、结果保存与历史回看能力不足。",
            "难以直接支撑毕业设计答辩中的完整展示流程。",
        ],
        COLORS["cyan"],
    )
    add_card(
        slide,
        430,
        126,
        482,
        104,
        "本课题目标",
        [
            "构建可演示、可管理、可归档的人脸替换系统。",
            "同步形成系统、论文与展示网站三项阶段成果。",
        ],
        COLORS["green_soft"],
        fill=COLORS["green_soft"],
    )
    add_card(
        slide,
        430,
        246,
        482,
        126,
        "中期汇报口径",
        [
            "课题属于系统实现型毕业设计，而非算法创新型研究。",
            "本次汇报重点展示工程闭环、真实材料与后续优化方向。",
        ],
        COLORS["yellow_soft"],
        fill=COLORS["white"],
    )

    add_textbox(slide, 48, 392, 180, 20, "从“能运行一次”到“能完整展示”", 12.5, COLORS["ink"], bold=True)
    chips = [
        ("算法调用原型", COLORS["blue_soft"]),
        ("桌面交互系统", COLORS["cyan_soft"]),
        ("结果回放与归档", COLORS["green_soft"]),
        ("论文与网站展示", COLORS["yellow_soft"]),
    ]
    x = 48
    for label, fill in chips:
        add_chip(slide, x, 420, 198, 40, label, fill)
        x += 216


def build_slide_3(slide):
    add_slide_bg(slide)
    add_title_block(slide, "阶段成果", "中期已完成成果总览", "系统、论文与展示网站三项成果已经形成可直接汇报的组合。", 3)
    add_round_box(slide, 48, 126, 864, 52, COLORS["navy"], line=None)
    add_textbox(slide, 72, 140, 816, 20, "中期阶段已具备“系统闭环 + 论文成稿 + 答辩展示页”的完整汇报基础。", 14, COLORS["white"], bold=True, align=ALIGN_CENTER)

    add_card(
        slide,
        48,
        206,
        264,
        204,
        "系统已完成主闭环",
        [
            "视频模式与摄像头模式均可演示。",
            "支持素材选择、参数配置与处理启动。",
            "处理结果可保存、回放并归档。",
        ],
        COLORS["cyan"],
        badge="01",
    )
    add_card(
        slide,
        348,
        206,
        264,
        204,
        "论文已形成完整结构",
        [
            "摘要、绪论、相关技术与需求分析已整理。",
            "总体设计、详细实现和测试分析已成型。",
            "论文内容与当前系统实现保持一致。",
        ],
        COLORS["green_soft"],
        badge="02",
    )
    add_card(
        slide,
        648,
        206,
        264,
        204,
        "展示网站已完成答辩页",
        [
            "页面覆盖功能、架构、界面、测试和展望。",
            "可作为现场讲解与成果汇总入口。",
            "整体口径与论文保持一致。",
        ],
        COLORS["yellow_soft"],
        badge="03",
    )
    add_textbox(slide, 48, 438, 864, 22, "当前重点是在已有闭环基础上继续补强细节、完善论文表达并优化答辩呈现。", 11.5, COLORS["muted"], align=ALIGN_CENTER)


def build_slide_4(slide):
    add_slide_bg(slide)
    add_title_block(slide, "技术路线", "技术路线与系统架构", "以“前端交互层 + 业务控制层 + 媒体处理层 + 数据持久化层”为主线组织系统。", 4)
    add_card(
        slide,
        48,
        126,
        272,
        246,
        "架构说明",
        [
            "PyQt5 负责界面组织、模式切换和状态反馈。",
            "线程层承担视频任务与摄像头任务调度。",
            "OpenCV / InsightFace 完成帧处理与换脸。",
            "Django REST / SQLite 负责素材与结果归档。",
        ],
        COLORS["cyan"],
    )

    add_round_box(slide, 344, 126, 568, 246, COLORS["white"], line=COLORS["line"])
    add_textbox(slide, 366, 142, 520, 20, "整体结构示意", 13, COLORS["ink"], bold=True)
    add_picture_fit(slide, ASSETS["architecture"], 362, 174, 532, 176)
    add_textbox(slide, 48, 404, 140, 18, "核心技术栈", 12.5, COLORS["ink"], bold=True)

    chip_specs = [
        ("PyQt5", COLORS["blue_soft"]),
        ("QThread", COLORS["cyan_soft"]),
        ("OpenCV", COLORS["green_soft"]),
        ("InsightFace", COLORS["yellow_soft"]),
        ("Django REST", COLORS["peach_soft"]),
        ("SQLite", COLORS["purple_soft"]),
    ]
    x = 48
    for label, fill in chip_specs:
        add_chip(slide, x, 432, 128, 34, label, fill)
        x += 140


def build_slide_5(slide):
    add_slide_bg(slide)
    add_title_block(slide, "系统功能", "系统核心功能", "围绕视频处理与实时演示，已形成素材进入系统到结果回放归档的完整闭环。", 5)

    cards = [
        (48, 132, "视频模式", ["导入输入视频与目标人脸。", "离线执行换脸并生成输出视频。"]),
        (336, 132, "摄像头模式", ["实时采集摄像头画面。", "支持现场演示与截图保存。"]),
        (624, 132, "素材管理", ["统一加载图片与视频素材。", "降低答辩时手工查找成本。"]),
        (192, 278, "结果回放", ["处理完成后直接预览输出结果。", "便于展示输入与输出差异。"]),
        (480, 278, "数据库归档", ["记录路径、时长、帧率和处理状态。", "为论文测试与结果留痕提供支撑。"]),
    ]
    accents = [COLORS["cyan"], COLORS["green_soft"], COLORS["yellow_soft"], COLORS["peach_soft"], COLORS["purple_soft"]]
    for idx, (left, top, title, lines) in enumerate(cards, start=1):
        add_card(slide, left, top, 240, 114, title, lines, accents[idx - 1], badge=f"{idx:02d}")

    add_round_box(slide, 48, 434, 864, 46, COLORS["navy"], line=None)
    add_textbox(slide, 72, 446, 816, 18, "素材选择 → 参数配置 → 处理执行 → 结果保存 → 回放归档", 13.5, COLORS["white"], bold=True, align=ALIGN_CENTER)


def build_slide_6(slide):
    add_slide_bg(slide)
    add_title_block(slide, "界面展示", "系统界面与处理效果", "展示内容全部来自当前项目的真实界面截图与样例图。", 6)

    add_round_box(slide, 48, 126, 520, 316, COLORS["white"], line=COLORS["line"])
    add_textbox(slide, 66, 142, 220, 18, "系统主界面", 13, COLORS["ink"], bold=True)
    add_picture_fit(slide, ASSETS["ui_main"], 62, 166, 492, 258)

    add_round_box(slide, 598, 126, 314, 190, COLORS["white"], line=COLORS["line"])
    add_textbox(slide, 616, 142, 220, 18, "样例 A：输入图像与换脸结果", 13, COLORS["ink"], bold=True)
    add_picture_fit(slide, ASSETS["pair_a"], 612, 166, 286, 134)

    add_card(
        slide,
        598,
        334,
        314,
        108,
        "答辩展示价值",
        [
            "主界面已集成素材选择、模式切换、参数控制与状态反馈。",
            "样例图能够直接说明系统已具备可视化换脸结果。",
        ],
        COLORS["cyan"],
    )
    add_round_box(slide, 48, 462, 864, 34, COLORS["cyan_soft"], line=None)
    add_textbox(slide, 64, 470, 832, 16, "系统可直接用于答辩现场演示：既能展示界面组织方式，也能展示真实处理效果。", 11, COLORS["navy"], bold=True, align=ALIGN_CENTER)


def build_slide_7(slide, temp_assets: dict[str, Path]):
    add_slide_bg(slide)
    add_title_block(slide, "展示网站", "展示网站成果", "网站承担“成果汇总 + 现场辅助讲解”的作用，页面内容与论文口径保持一致。", 7)

    add_browser_frame(slide, 48, 126, 336, 332, temp_assets["website_top"])
    add_textbox(slide, 48, 468, 336, 18, "答辩展示页真实渲染截图", 10.5, COLORS["muted"], align=ALIGN_CENTER)

    add_card(
        slide,
        420,
        126,
        492,
        126,
        "网站定位",
        [
            "用于集中展示系统背景、功能、技术路线、界面与测试分析。",
            "适合作为答辩讲解时的辅助页面，也便于成果统一汇总。",
        ],
        COLORS["green_soft"],
    )
    add_textbox(slide, 420, 286, 180, 18, "页面主要章节", 12.5, COLORS["ink"], bold=True)
    chips = [
        ("系统功能", COLORS["blue_soft"]),
        ("技术路线", COLORS["cyan_soft"]),
        ("界面展示", COLORS["green_soft"]),
        ("测试分析", COLORS["yellow_soft"]),
        ("总结展望", COLORS["peach_soft"]),
    ]
    positions = [(420, 314), (578, 314), (736, 314), (500, 360), (658, 360)]
    for (label, fill), (x, y) in zip(chips, positions):
        add_chip(slide, x, y, 142, 36, label, fill)
    add_round_box(slide, 420, 414, 492, 44, COLORS["white"], line=COLORS["line"])
    add_textbox(slide, 442, 426, 448, 16, "页面内容与系统、论文使用同一套阶段成果材料，不使用宣传式夸张表述。", 10.5, COLORS["muted"], align=ALIGN_CENTER)


def build_slide_8(slide):
    add_slide_bg(slide)
    add_title_block(slide, "论文进度", "论文完成情况", "论文已围绕系统实现与工程验证形成完整章节结构，并与系统和网站内容对齐。", 8)

    chapters = [
        ("01", "摘要与关键词"),
        ("02", "绪论"),
        ("03", "相关技术"),
        ("04", "需求分析"),
        ("05", "总体设计"),
        ("06", "详细实现"),
        ("07", "测试分析"),
        ("08", "总结展望"),
    ]
    positions = [
        (48, 126), (300, 126),
        (48, 208), (300, 208),
        (48, 290), (300, 290),
        (48, 372), (300, 372),
    ]
    fills = [COLORS["blue_soft"], COLORS["cyan_soft"], COLORS["green_soft"], COLORS["yellow_soft"], COLORS["peach_soft"], COLORS["purple_soft"], COLORS["blue_soft"], COLORS["cyan_soft"]]
    for (num, label), (x, y), fill in zip(chapters, positions, fills):
        add_round_box(slide, x, y, 224, 64, fill, line=COLORS["line"])
        add_textbox(slide, x + 16, y + 12, 40, 20, num, 10.5, COLORS["muted"], bold=True, font_name="Segoe UI")
        add_textbox(slide, x + 16, y + 28, 190, 20, label, 15, COLORS["ink"], bold=True)

    add_card(
        slide,
        564,
        126,
        348,
        182,
        "当前完成状态",
        [
            "论文题目、中英文摘要与目录结构已完善。",
            "需求分析、总体设计、详细实现与测试分析均已写入正文。",
            "使用真实数据库记录、截图与样例图支撑论述。",
        ],
        COLORS["cyan"],
    )
    add_card(
        slide,
        564,
        328,
        348,
        108,
        "论文定位",
        [
            "不是算法创新型论文，重点展示系统实现与工程验证。",
            "测试口径与展示页统一采用 2026 年 4 月 2 日本地项目数据。",
        ],
        COLORS["yellow_soft"],
    )
    add_round_box(slide, 564, 452, 348, 34, COLORS["navy"], line=None)
    add_textbox(slide, 580, 460, 316, 16, "论文、系统与网站三者内容已对齐，适合中期答辩使用。", 10.5, COLORS["white"], bold=True, align=ALIGN_CENTER)


def build_slide_9(slide):
    add_slide_bg(slide)
    add_title_block(slide, "测试分析", "测试分析与阶段结论", "统计口径统一采用论文和展示页中已使用的 2026 年 4 月 2 日本地项目数据。", 9)

    add_metric_card(slide, 48, 126, 206, 104, "人脸图片记录", "19", COLORS["cyan"], "本地有效路径：9 / 9")
    add_metric_card(slide, 274, 126, 206, 104, "输入视频记录", "13", COLORS["peach_soft"], "本地有效路径：7 / 8")
    add_metric_card(slide, 500, 126, 206, 104, "输出视频记录", "5", COLORS["green_soft"], "输出目录中实际有 10 个 mp4")
    add_metric_card(slide, 726, 126, 186, 104, "阶段判断", "主闭环\n已形成", COLORS["yellow_soft"], "强调工程可用性", value_size=19, note_size=10)

    add_bar_chart(slide, 48, 262, 384, 186)
    add_card(
        slide,
        458,
        262,
        454,
        186,
        "阶段结论",
        [
            "系统已经能够跑通素材加载、处理执行、结果保存和回放验证。",
            "真实数据说明主流程可用，但路径治理与结果补登记仍需加强。",
            "中期答辩更强调工程闭环和问题分析，而非追求大规模算法对照实验。",
        ],
        COLORS["cyan"],
    )
    add_round_box(slide, 48, 466, 864, 32, COLORS["cyan_soft"], line=None)
    add_textbox(slide, 64, 473, 832, 14, "当前系统已经具备“可展示、可管理、可分析”的阶段性成果基础。", 10.5, COLORS["navy"], bold=True, align=ALIGN_CENTER)


def build_slide_10(slide):
    add_slide_bg(slide)
    add_title_block(slide, "后续安排", "当前问题与后续计划", "中期答辩需要主动说明现存不足，并给出清晰的后续补强方向。", 10)

    add_card(
        slide,
        48,
        126,
        398,
        304,
        "当前仍需完善的问题",
        [
            "历史资源路径治理仍需继续收敛。",
            "输出结果与数据库记录之间还存在补登记不足。",
            "processing_tasks 尚未形成真正的排队与调度机制。",
            "日志展示与异常恢复仍然偏简化。",
        ],
        COLORS["peach_soft"],
        fill=COLORS["red_soft"],
    )
    add_card(
        slide,
        514,
        126,
        398,
        304,
        "下一步计划",
        [
            "统一媒体目录与相对路径策略，减少历史路径失效。",
            "增加输出目录扫描与补登记逻辑，完善结果回补。",
            "继续完善任务状态、取消与排队机制。",
            "优化论文细节、补充答辩讲稿并继续打磨展示材料。",
        ],
        COLORS["cyan"],
        fill=COLORS["cyan_soft"],
    )
    add_round_box(slide, 48, 452, 864, 40, COLORS["navy"], line=None)
    add_textbox(slide, 64, 462, 832, 18, "中期结论：系统、论文与网站三项成果已经具备答辩展示基础，后续重点转向补强与收尾。", 11.5, COLORS["white"], bold=True, align=ALIGN_CENTER)


def build_slide_11(slide):
    add_slide_bg(slide, dark=True)
    add_textbox(slide, 0, 158, 960, 52, "谢谢聆听", 28, COLORS["white"], bold=True, align=ALIGN_CENTER)
    add_textbox(slide, 0, 220, 960, 26, "欢迎老师批评指正", 14, COLORS["white"], align=ALIGN_CENTER)
    add_round_box(slide, 296, 286, 368, 38, COLORS["navy_soft"], line=None, transparency=0.18)
    add_textbox(slide, 316, 295, 328, 18, "人脸替换系统的设计与实现 | 中期答辩", 11, COLORS["white"], bold=True, align=ALIGN_CENTER)


def build_presentation():
    temp_assets = prepare_temp_assets()
    safe_unlink(OUT_PPTX)
    safe_unlink(OUT_PDF)
    shutil.rmtree(PREVIEW_DIR, ignore_errors=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    app = win32com.client.Dispatch("PowerPoint.Application")
    app.Visible = True
    app.DisplayAlerts = 0
    presentation = app.Presentations.Add()

    try:
        presentation.PageSetup.SlideWidth = SLIDE_W
        presentation.PageSetup.SlideHeight = SLIDE_H

        slides = []
        for _ in range(11):
            slides.append(presentation.Slides.Add(presentation.Slides.Count + 1, PP_LAYOUT_BLANK))

        build_cover(slides[0])
        build_slide_2(slides[1])
        build_slide_3(slides[2])
        build_slide_4(slides[3])
        build_slide_5(slides[4])
        build_slide_6(slides[5])
        build_slide_7(slides[6], temp_assets)
        build_slide_8(slides[7])
        build_slide_9(slides[8])
        build_slide_10(slides[9])
        build_slide_11(slides[10])

        presentation.SaveAs(str(OUT_PPTX), PPTX_FORMAT)
        presentation.SaveAs(str(OUT_PDF), PDF_FORMAT)
        presentation.Export(str(PREVIEW_DIR), "PNG", 1600, 900)
        print(f"PPTX: {OUT_PPTX}")
        print(f"PDF: {OUT_PDF}")
        print(f"PreviewDir: {PREVIEW_DIR}")
    finally:
        presentation.Close()
        app.Quit()


if __name__ == "__main__":
    build_presentation()
