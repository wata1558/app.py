from ultralytics import YOLO

# PyTorch モデル読み込み
model = YOLO("best.pt")
print(model.names)

# ONNX 変換
model.export(
    format="onnx",   # 出力形式
    opset=12,        # 推奨 opset バージョン
    dynamic=False,   # 画像サイズ固定にする（訓練時と同じサイズに揃える）
    simplify=True    # ONNX simplifier を使って出力を整理
)
