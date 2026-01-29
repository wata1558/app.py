from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
from ultralytics import YOLO
import traceback
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Static Image YOLO Detection")

# ONNXモデルをロード
model = YOLO("best.onnx")  # best.onnxを使用

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://heatmap-7a032.web.app"],  # Flutter Web アプリの URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

label_map = {
    "can": "缶",
    "cigarette": "タバコ",
    "paper": "紙",
    "plastic": "ペットボトル",
    "plasticbag": "袋"
}

@app.get("/")
def root():
    return {"message": "Send a POST request to /detect with an image file."}

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    try:
        # 画像を読み込み
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((320, 320))
        img_np = np.array(img)

        # ONNXモデルで推論
        results = model.predict(img_np, conf=0.4, iou=0.3, device="cpu")
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = label_map.get(model.names[cls_id], model.names[cls_id])
                detections.append({
                    "label": label,
                    "confidence": round(conf, 2)
                })

        return JSONResponse(content={"detections": detections})

    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "trace": traceback.format_exc()},
            status_code=500
        )
