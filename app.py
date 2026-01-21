from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
from ultralytics import YOLO
import traceback
import torch
from torch.quantization import quantize_dynamic

app = FastAPI(title="Static Image YOLO Detection")

model = YOLO("best32.pt")  # Render にモデルがあることを確認

model.model = quantize_dynamic(
    model.model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

label_map = {
    "can": "缶",
    "cigarette": "タバコ",
    "paper": "紙",
    "plastic": "ペットボトル",  # plasticはAPIではペットボトルとして返す
    "plasticbag": "袋"
}

@app.get("/")
def root():
    return {"message": "Send a POST request to /detect with an image file."}

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((320, 320))
        img_np = np.array(img)

        results = model.predict(img_np, conf=0.3, iou=0.3, device="cpu")
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                label = label_map.get(model.names[cls_id], model.names[cls_id])
                detections.append({
                    "label": label,
                })

        return JSONResponse(content={"detections": detections})

    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "trace": traceback.format_exc()},
            status_code=500
        )
