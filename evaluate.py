import cv2
import pandas as pd
from ultralytics import YOLO

from src import config
from src.metrics import match_frame
from src.utils import resize_width

gt = pd.read_csv("data/ground_truth.csv")
model = YOLO(config.YOLO_MODEL)
cap = cv2.VideoCapture(config.VIDEO_PATH)

TP = FP = FN = 0
for frame_idx in sorted(gt["frame"].unique()):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    if not ret:
        continue
    frame = resize_width(frame, config.FRAME_WIDTH)

    r = model.predict(frame, conf=config.YOLO_CONF, verbose=False)[0]
    preds = [{"name": r.names[int(c)], "conf": float(s),
              "box": tuple(b.tolist())}
             for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
    gts = [{"name": row["name"], "box": (row.x1, row.y1, row.x2, row.y2)}
           for _, row in gt[gt["frame"] == frame_idx].iterrows()]

    tp, fp, fn = match_frame(preds, gts, iou_thr=0.5)
    TP, FP, FN = TP + tp, FP + fp, FN + fn

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
print(f"Precision {precision:.3f} | Recall {recall:.3f} | F1 {f1:.3f}")