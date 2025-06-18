from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import io
import base64

app = FastAPI()

@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # 1. Hafif parlaklık artır
    brightness_enhancer = ImageEnhance.Brightness(img)
    img = brightness_enhancer.enhance(1.15)  # %15 artır

    # 2. Renk doygunluğu artır
    color_enhancer = ImageEnhance.Color(img)
    img = color_enhancer.enhance(1.25)  # %25 daha canlı renkler

    # 3. Açı düzelt
    img_np = np.array(img)
    angle = detect_skew_angle(img_np)
    img_rotated = rotate_image(img_np, angle)

    # 4. Çözünürlük kontrolü
    result_img = Image.fromarray(img_rotated).resize((800, 600))

    buffer = io.BytesIO()
    result_img.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return JSONResponse(content={"image_base64": img_base64})

def detect_skew_angle(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    angles = [(theta * 180 / np.pi) - 90 for rho, theta in lines[:, 0]] if lines is not None else []
    return -np.median(angles) if angles else 0

def rotate_image(img_np, angle):
    (h, w) = img_np.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img_np, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
