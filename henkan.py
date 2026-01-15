from ultralytics import YOLO
model = YOLO("best32.pt")
model.export(format="onnx", dynamic=True, simplify=True)
