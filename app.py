from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
from ultralytics import YOLO
import traceback

app = FastAPI(title="Static Image YOLO Detection")

model = YOLO("best32.pt")  # Render にモデルがあることを確認

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
        img_np = np.array(img)

        results = model.predict(img_np, conf=0.6, iou=0.3, device="cpu")
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                label = model.names[cls_id]
                xyxy = box.xyxy.cpu().numpy().tolist()[0]
                confidence = float(box.conf.cpu().numpy()[0])
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": xyxy
                })

        return JSONResponse(content={"detections": detections})

    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "trace": traceback.format_exc()},
            status_code=500
        )
