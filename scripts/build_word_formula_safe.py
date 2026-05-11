from __future__ import annotations

import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH


ROOT = Path(r"E:\face")
SRC_DOCX = ROOT / "换脸系统_初稿_学术化版_终稿统一整改版_图位公式调整版.docx"
SRC_MD = ROOT / "换脸系统_初稿_学术化版_终稿统一整改版_图位公式调整版.md"
DST_DOCX = ROOT / "换脸系统_初稿_学术化版_终稿统一整改版_Word公式安全版.docx"
DST_MD = ROOT / "换脸系统_初稿_学术化版_终稿统一整改版_Word公式安全版.md"


DOCX_BLOCKS = {
    "若将源人脸关键点集合记为 P_s = {p_i^s}_{i=1}^n，目标人脸关键点集合记为 P_t = {p_i^t}_{i=1}^n，则传统三角剖分换脸通常先基于关键点进行 Delaunay 三角剖分，得到三角形集合 T = {tau_k}_{k=1}^m，并在源图与目标图之间建立逐三角形对应关系。对于第 k 个对应三角形，其局部几何映射一般可表示为 [x_t, y_t]^T = A_k [x_s, y_s]^T + b_k，其中 A_k 表示 2 x 2 仿射变换矩阵，b_k 表示平移向量。该表达式反映了传统方法将整体换脸问题分解为多个局部线性形变子问题的基本思想。": [
        ("text", "若将源人脸关键点集合与目标人脸关键点集合分别记为"),
        ("formula", "P_s = {p_i^s}_{i = 1}^n"),
        ("formula", "P_t = {p_i^t}_{i = 1}^n"),
        ("text", "则传统三角剖分换脸通常先基于关键点进行 Delaunay 三角剖分，得到三角形集合"),
        ("formula", r"T = {\tau_k}_{k = 1}^m"),
        ("text", "并在源图与目标图之间建立逐三角形对应关系。对于第 k 个对应三角形，其局部几何映射可进一步写为"),
        ("formula", r"x_t = a_(11)^(k) x_s + a_(12)^(k) y_s + b_1^(k)"),
        ("formula", r"y_t = a_(21)^(k) x_s + a_(22)^(k) y_s + b_2^(k)"),
        ("text", "其中，a_(11)^(k)、a_(12)^(k)、a_(21)^(k) 与 a_(22)^(k) 共同刻画第 k 个三角区域的局部仿射变换系数，b_1^(k) 与 b_2^(k) 表示平移项。上述表达反映了传统方法将整体换脸问题分解为多个局部线性形变子问题的基本思想。"),
    ],
    "在具体像素重映射过程中，若目标三角形内部像素 p 满足 p = lambda_1 v_1^t + lambda_2 v_2^t + lambda_3 v_3^t，且 lambda_1 + lambda_2 + lambda_3 = 1、lambda_i >= 0，则可利用同一组重心坐标在源三角形中定位对应像素 q = lambda_1 v_1^s + lambda_2 v_2^s + lambda_3 v_3^s，并完成纹理采样。完成局部仿射变换后，基础融合常写为 I_out(p) = M(p) I_warp(p) + (1 - M(p)) I_t(p)，其中 M(p) 表示融合掩模，I_warp(p) 表示变形后的人脸纹理，I_t(p) 表示目标图像像素。若进一步采用无缝融合，则可通过最小化 int_Omega ||grad f - grad I_warp||^2 dOmega 来减弱边界拼接痕迹，这也是传统方法常见的后处理思路[1][2]。": [
        ("text", "在具体像素重映射过程中，若目标三角形内部像素 p 满足"),
        ("formula", r"p = \lambda_1 v_1^t + \lambda_2 v_2^t + \lambda_3 v_3^t"),
        ("formula", r"\lambda_1 + \lambda_2 + \lambda_3 = 1, \lambda_i \ge 0"),
        ("text", "则可利用同一组重心坐标在源三角形中定位对应像素"),
        ("formula", r"q = \lambda_1 v_1^s + \lambda_2 v_2^s + \lambda_3 v_3^s"),
        ("text", "并完成纹理采样。完成局部仿射变换后，基础融合可写为"),
        ("formula", r"I_(out)(p) = M(p) I_(warp)(p) + (1 - M(p)) I_t(p)"),
        ("text", "其中，M(p) 表示融合掩模，I_(warp)(p) 表示变形后的人脸纹理，I_t(p) 表示目标图像像素。若进一步采用无缝融合，则可通过最小化下式减弱边界拼接痕迹"),
        ("formula", r"min \int_\Omega ||\nabla f - \nabla I_(warp)||^2 d\Omega"),
        ("text", "这也是传统方法中较为常见的后处理思路[1][2]。"),
    ],
    "基于深度学习的人脸替换可抽象为身份特征提取与目标属性保持的联合映射过程。设源人脸图像为 I_s，目标图像或目标帧为 I_t，身份编码器 F_id 从源图中提取身份嵌入 e_s = F_id(I_s)，随后由生成或替换网络给出输出 I_hat = G(I_t, e_s)。在该表达式中，e_s 主要承担身份信息约束，而 I_t 中的姿态、表情、光照与背景结构则作为目标属性被尽量保留。因此，深度学习方法的关键不再只是局部几何对齐，而是如何在高维特征空间内同时协调身份一致性与目标场景一致性。": [
        ("text", "基于深度学习的人脸替换可抽象为身份特征提取与目标属性保持的联合映射过程。设源人脸图像为 I_s，目标图像或目标帧为 I_t，则身份编码与生成过程可写为"),
        ("formula", r"e_s = F_(id)(I_s)"),
        ("formula", r"I_(hat) = G(I_t, e_s)"),
        ("text", "在上述表达式中，e_s 主要承担身份信息约束，而 I_t 中的姿态、表情、光照与背景结构则作为目标属性被尽量保留。因此，深度学习方法的关键不再只是局部几何对齐，而是如何在高维特征空间内同时协调身份一致性与目标场景一致性。"),
    ],
    "从训练目标看，此类方法通常将总损失写为 L = lambda_id L_id + lambda_rec L_rec + lambda_per L_per + lambda_adv L_adv。其中，身份保持项可表示为 L_id = 1 - cos(E(I_hat), E(I_s))，E(.) 表示用于身份判别的嵌入网络；L_rec 用于约束内容重建或结构一致性，L_per 用于衡量高层感知差异，L_adv 则用于提升结果的真实感与分布一致性。就本文所使用的 InsightFace 方案而言，其工程优势在于能够直接调用预训练身份表征与换脸能力，在不重新训练系统模型的前提下完成较高质量的人脸身份迁移[4][6][9][10][11][12]。": [
        ("text", "从训练目标看，此类方法通常将总损失写为"),
        ("formula", r"L = \lambda_(id) L_(id) + \lambda_(rec) L_(rec) + \lambda_(per) L_(per) + \lambda_(adv) L_(adv)"),
        ("text", "其中，身份保持项可进一步表示为"),
        ("formula", r"L_(id) = 1 - cos(E(I_(hat)), E(I_s))"),
        ("text", "E(.) 表示用于身份判别的嵌入网络；L_(rec) 用于约束内容重建或结构一致性，L_(per) 用于衡量高层感知差异，L_(adv) 则用于提升结果的真实感与分布一致性。就本文所使用的 InsightFace 方案而言，其工程优势在于能够直接调用预训练身份表征与换脸能力，在不重新训练系统模型的前提下完成较高质量的人脸身份迁移[4][6][9][10][11][12]。"),
    ],
}


MD_REPLACEMENTS = {
    "若将源人脸关键点集合记为 P_s = {p_i^s}_{i=1}^n，目标人脸关键点集合记为 P_t = {p_i^t}_{i=1}^n，则传统三角剖分换脸通常先基于关键点进行 Delaunay 三角剖分，得到三角形集合 T = {tau_k}_{k=1}^m，并在源图与目标图之间建立逐三角形对应关系。对于第 k 个对应三角形，其局部几何映射一般可表示为 [x_t, y_t]^T = A_k [x_s, y_s]^T + b_k，其中 A_k 表示 2 x 2 仿射变换矩阵，b_k 表示平移向量。该表达式反映了传统方法将整体换脸问题分解为多个局部线性形变子问题的基本思想。": """若将源人脸关键点集合与目标人脸关键点集合分别记为

P_s = {p_i^s}_{i = 1}^n

P_t = {p_i^t}_{i = 1}^n

则传统三角剖分换脸通常先基于关键点进行 Delaunay 三角剖分，得到三角形集合

T = {\\tau_k}_{k = 1}^m

并在源图与目标图之间建立逐三角形对应关系。对于第 k 个对应三角形，其局部几何映射可进一步写为

x_t = a_(11)^(k) x_s + a_(12)^(k) y_s + b_1^(k)

y_t = a_(21)^(k) x_s + a_(22)^(k) y_s + b_2^(k)

其中，a_(11)^(k)、a_(12)^(k)、a_(21)^(k) 与 a_(22)^(k) 共同刻画第 k 个三角区域的局部仿射变换系数，b_1^(k) 与 b_2^(k) 表示平移项。上述表达反映了传统方法将整体换脸问题分解为多个局部线性形变子问题的基本思想。""",
    "在具体像素重映射过程中，若目标三角形内部像素 p 满足 p = lambda_1 v_1^t + lambda_2 v_2^t + lambda_3 v_3^t，且 lambda_1 + lambda_2 + lambda_3 = 1、lambda_i >= 0，则可利用同一组重心坐标在源三角形中定位对应像素 q = lambda_1 v_1^s + lambda_2 v_2^s + lambda_3 v_3^s，并完成纹理采样。完成局部仿射变换后，基础融合常写为 I_out(p) = M(p) I_warp(p) + (1 - M(p)) I_t(p)，其中 M(p) 表示融合掩模，I_warp(p) 表示变形后的人脸纹理，I_t(p) 表示目标图像像素。若进一步采用无缝融合，则可通过最小化 int_Omega ||grad f - grad I_warp||^2 dOmega 来减弱边界拼接痕迹，这也是传统方法常见的后处理思路[1][2]。": """在具体像素重映射过程中，若目标三角形内部像素 p 满足

p = \\lambda_1 v_1^t + \\lambda_2 v_2^t + \\lambda_3 v_3^t

\\lambda_1 + \\lambda_2 + \\lambda_3 = 1, \\lambda_i \\ge 0

则可利用同一组重心坐标在源三角形中定位对应像素

q = \\lambda_1 v_1^s + \\lambda_2 v_2^s + \\lambda_3 v_3^s

并完成纹理采样。完成局部仿射变换后，基础融合可写为

I_(out)(p) = M(p) I_(warp)(p) + (1 - M(p)) I_t(p)

其中，M(p) 表示融合掩模，I_(warp)(p) 表示变形后的人脸纹理，I_t(p) 表示目标图像像素。若进一步采用无缝融合，则可通过最小化下式减弱边界拼接痕迹

min \\int_\\Omega ||\\nabla f - \\nabla I_(warp)||^2 d\\Omega

这也是传统方法中较为常见的后处理思路[1][2]。""",
    "基于深度学习的人脸替换可抽象为身份特征提取与目标属性保持的联合映射过程。设源人脸图像为 I_s，目标图像或目标帧为 I_t，身份编码器 F_id 从源图中提取身份嵌入 e_s = F_id(I_s)，随后由生成或替换网络给出输出 I_hat = G(I_t, e_s)。在该表达式中，e_s 主要承担身份信息约束，而 I_t 中的姿态、表情、光照与背景结构则作为目标属性被尽量保留。因此，深度学习方法的关键不再只是局部几何对齐，而是如何在高维特征空间内同时协调身份一致性与目标场景一致性。": """基于深度学习的人脸替换可抽象为身份特征提取与目标属性保持的联合映射过程。设源人脸图像为 I_s，目标图像或目标帧为 I_t，则身份编码与生成过程可写为

e_s = F_(id)(I_s)

I_(hat) = G(I_t, e_s)

在上述表达式中，e_s 主要承担身份信息约束，而 I_t 中的姿态、表情、光照与背景结构则作为目标属性被尽量保留。因此，深度学习方法的关键不再只是局部几何对齐，而是如何在高维特征空间内同时协调身份一致性与目标场景一致性。""",
    "从训练目标看，此类方法通常将总损失写为 L = lambda_id L_id + lambda_rec L_rec + lambda_per L_per + lambda_adv L_adv。其中，身份保持项可表示为 L_id = 1 - cos(E(I_hat), E(I_s))，E(.) 表示用于身份判别的嵌入网络；L_rec 用于约束内容重建或结构一致性，L_per 用于衡量高层感知差异，L_adv 则用于提升结果的真实感与分布一致性。就本文所使用的 InsightFace 方案而言，其工程优势在于能够直接调用预训练身份表征与换脸能力，在不重新训练系统模型的前提下完成较高质量的人脸身份迁移[4][6][9][10][11][12]。": """从训练目标看，此类方法通常将总损失写为

L = \\lambda_(id) L_(id) + \\lambda_(rec) L_(rec) + \\lambda_(per) L_(per) + \\lambda_(adv) L_(adv)

其中，身份保持项可进一步表示为

L_(id) = 1 - cos(E(I_(hat)), E(I_s))

E(.) 表示用于身份判别的嵌入网络；L_(rec) 用于约束内容重建或结构一致性，L_(per) 用于衡量高层感知差异，L_(adv) 则用于提升结果的真实感与分布一致性。就本文所使用的 InsightFace 方案而言，其工程优势在于能够直接调用预训练身份表征与换脸能力，在不重新训练系统模型的前提下完成较高质量的人脸身份迁移[4][6][9][10][11][12]。""",
}


def set_formula_paragraph(paragraph, text: str) -> None:
    paragraph.text = text
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


def set_text_paragraph(paragraph, text: str) -> None:
    paragraph.text = text
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def transform_docx() -> None:
    shutil.copyfile(SRC_DOCX, DST_DOCX)
    doc = Document(str(DST_DOCX))

    for paragraph in list(doc.paragraphs):
        block = DOCX_BLOCKS.get(paragraph.text.strip())
        if not block:
            continue

        style = paragraph.style
        last_type, last_text = block[-1]
        if last_type == "formula":
            set_formula_paragraph(paragraph, last_text)
        else:
            set_text_paragraph(paragraph, last_text)
        paragraph.style = style

        for item_type, item_text in reversed(block[:-1]):
            inserted = paragraph.insert_paragraph_before(item_text, style=style)
            if item_type == "formula":
                inserted.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                inserted.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    doc.save(str(DST_DOCX))


def transform_md() -> None:
    shutil.copyfile(SRC_MD, DST_MD)
    text = DST_MD.read_text(encoding="utf-8")
    for old, new in MD_REPLACEMENTS.items():
        text = text.replace(old, new)
    DST_MD.write_text(text, encoding="utf-8")


def main() -> None:
    transform_docx()
    transform_md()
    print(DST_DOCX)
    print(DST_MD)


if __name__ == "__main__":
    main()
