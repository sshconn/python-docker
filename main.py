from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import base64
import io

app = FastAPI()

@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    input_image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Görsel iyileştirme
    enhancer = ImageEnhance.Brightness(input_image)
    image = enhancer.enhance(1.1)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.1)

    # Açı düzeltme
    image_np = np.array(image)
    angle = detect_skew_angle(image_np)
    rotated = rotate_image(image_np, angle)
    result = Image.fromarray(rotated)

    # Görseli base64 olarak döndür
    buffered = io.BytesIO()
    result.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return JSONResponse(content={"image_base64": img_str})

def detect_skew_angle(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    angles = [(theta * 180 / np.pi) - 90 for rho, theta in lines[:, 0]] if lines is not None else []
    return -np.median(angles) if angles else 0

def rotate_image(image_np, angle):
    (h, w) = image_np.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image_np, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
