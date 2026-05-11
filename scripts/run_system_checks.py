#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Engineering checks for the face-swap defense demo."""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "backend-django" / "db.sqlite3"
MODEL_DIR = ROOT / "models"


def ok(name: str, passed: bool, detail: str = "") -> bool:
    marker = "PASS" if passed else "FAIL"
    print(f"[{marker}] {name}: {detail}")
    return passed


def check_database() -> bool:
    if not DB_PATH.exists():
        return ok("database", False, f"missing {DB_PATH}")
    con = sqlite3.connect(DB_PATH)
    try:
        counts = {}
        for table in ["face_images", "input_videos", "output_videos"]:
            counts[table] = con.execute(f"select count(*) from {table}").fetchone()[0]
        return ok("database records", True, str(counts))
    finally:
        con.close()


def check_cuda() -> bool:
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        return ok("CUDA provider", "CUDAExecutionProvider" in providers, str(providers))
    except Exception as exc:
        return ok("CUDA provider", False, str(exc))


def check_models() -> bool:
    required = [
        MODEL_DIR / "inswapper_128.onnx",
        MODEL_DIR / "haarcascade_frontalface_default.xml",
        MODEL_DIR / "shape_predictor_68_face_landmarks.dat",
        MODEL_DIR / "models" / "buffalo_l",
    ]
    missing = [str(path) for path in required if not path.exists()]
    return ok("model files", not missing, "missing: " + ", ".join(missing) if missing else "all present")


def check_camera() -> bool:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    opened = cap.isOpened()
    detail = "opened" if opened else "not available"
    cap.release()
    return ok("camera", opened, detail)


def check_chinese_path_image() -> bool:
    temp_dir = Path(tempfile.gettempdir()) / "中文路径测试"
    temp_dir.mkdir(exist_ok=True)
    image_path = temp_dir / "测试图片.png"
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :] = (0, 128, 255)
    encoded = cv2.imencode(".png", image)[1]
    encoded.tofile(str(image_path))
    decoded = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    return ok("Chinese path image read", decoded is not None and decoded.shape[:2] == (48, 64), str(image_path))


def check_upload_dir() -> bool:
    upload_dir = ROOT / "data" / "input_faces"
    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
        probe = upload_dir / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return ok("upload directory writable", True, str(upload_dir))
    except Exception as exc:
        return ok("upload directory writable", False, str(exc))


def main() -> int:
    checks = [
        check_database(),
        check_cuda(),
        check_models(),
        check_camera(),
        check_chinese_path_image(),
        check_upload_dir(),
    ]
    passed = sum(1 for item in checks if item)
    print(f"\nSummary: {passed}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
