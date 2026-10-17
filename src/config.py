# ---------- Input / output ----------
VIDEO_PATH = "data/test.mp4"
OUTPUT_VIDEO = "outputs/videos/result.mp4"
LOG_DIR = "outputs/logs"
PLOT_DIR = "outputs/plots"

# ---------- Frame ----------
FRAME_WIDTH = 640            # resize all frames to this width

# ---------- Classical pipeline ----------
BLUR_KERNEL = (5, 5)         # must be odd numbers
BLUR_SIGMA = 0               # 0 = OpenCV computes sigma from kernel size
ADAPT_BLOCK = 11             # neighborhood size for adaptive threshold (odd)
ADAPT_C = 2                  # constant subtracted from the local mean
MORPH_KERNEL = (3, 3)
MIN_AREA = 500               # "pixel bounds": ignore contours smaller than this
MAX_AREA = 50000             # ...and larger than this

# ---------- YOLO ----------
YOLO_MODEL = "yolov8n.pt"    # n = nano (fastest). Try s, m for experiments
YOLO_CONF = 0.4              # minimum confidence
TARGET_CLASSES = None        # e.g. [0, 2] = person, car. None = all 80 COCO classes