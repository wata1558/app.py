from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
from ultralytics import YOLO

app = FastAPI(title="Static Image YOLO Detection")

# 学習済みモデル読み込み
model = YOLO("best32.pt")  # あなたのモデルパスに変更

@app.get("/")
def root():
    return {"message": "Send a POST request to /detect with an image file."}

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    try:
        # 画像を読み込み
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 推論
        results = model.predict(img, conf=0.6, iou=0.3)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                label = model.names[cls_id]
                xyxy = box.xyxy.cpu().numpy().tolist()[0]  # [x1, y1, x2, y2]
                detections.append({
                    "label": label,
                    "confidence": float(box.conf.cpu().numpy()[0]),
                    "bbox": xyxy
                })

        return JSONResponse(content={"detections": detections})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
