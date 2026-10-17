import pandas as pd

from src import config
from src.pipeline import process_video

results = []
common = dict(show=False, max_frames=300)

# Experiment 1: Gaussian kernel size (classical only)
for k in [3, 5, 9, 15, 25]:
    s = process_video(use_yolo=False, blur_k=(k, k), **common).summary()
    results.append({"experiment": "kernel", "setting": k, **s})

# Experiment 2: YOLO model size
for m in ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]:
    s = process_video(use_classical=False, model_name=m, **common).summary()
    results.append({"experiment": "model", "setting": m, **s})

# Experiment 3: frame resolution (YOLO nano)
for w in [320, 480, 640, 960]:
    s = process_video(use_classical=False, width=w, **common).summary()
    results.append({"experiment": "resolution", "setting": w, **s})

# Experiment 4: classical vs YOLO vs both
for mode, (c, y) in {"classical": (True, False), "yolo": (False, True),
                     "both": (True, True)}.items():
    s = process_video(use_classical=c, use_yolo=y, **common).summary()
    results.append({"experiment": "mode", "setting": mode, **s})

df = pd.DataFrame(results)
df.to_csv(f"{config.LOG_DIR}/experiments.csv", index=False)
print(df)