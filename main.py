from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageEnhance
import numpy as np
import io, base64
import cv2

app = FastAPI()

@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    input_image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Görsel iyileştirme
    enhancer = ImageEnhance.Brightness(input_image)
    image = enhancer.enhance(1.1)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.1)

    # Eğim düzeltme
    image_np = np.array(image)
    angle = detect_skew_angle(image_np)
    result_np = rotate_image(image_np, angle)
    result_image = Image.fromarray(result_np)

    buffer = io.BytesIO()
    result_image.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return JSONResponse(content={"image_base64": img_base64})

# Skew düzeltme fonksiyonları aynen kullanılabilir


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
