import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Static Image YOLO Detection")

# ONNXモデルをロード
try:
    model = YOLO("best.onnx", task="detect")
    logging.info("ONNXモデルロード成功")
except Exception as e:
    logging.exception("ONNXモデルロード失敗")
    raise e

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://heatmap-7a032.web.app"],
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
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)

        logging.info(f"入力画像形状: {img_np.shape}")
        logging.info(f"model.names: {model.names}")


        results = model.predict(img_np, conf=0.5, iou=0.3, device="cpu", verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                logging.info(f"検出されたcls_id: {cls_id}, confidence: {conf}")
                if cls_id in model.names:
                     label = label_map.get(model.names[cls_id], model.names[cls_id])
                else:
                    label = "その他"
                detections.append({"label": label, "confidence": round(conf, 2)})

        logging.info(f"検出結果: {detections}")
        return JSONResponse(content={"detections": detections})

    except Exception as e:
        logging.exception("推論中にエラー発生")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
