import argparse

from src import config
from src.pipeline import process_video

parser = argparse.ArgumentParser()
parser.add_argument("--video", default=config.VIDEO_PATH)
parser.add_argument("--mode", choices=["classical", "yolo", "both"], default="both")
parser.add_argument("--model", default=config.YOLO_MODEL)
parser.add_argument("--no-show", action="store_true")
parser.add_argument("--save", action="store_true")
args = parser.parse_args()

logger = process_video(
    video_path=args.video,
    use_classical=args.mode in ("classical", "both"),
    use_yolo=args.mode in ("yolo", "both"),
    model_name=args.model,
    show=not args.no_show,
    save_path=config.OUTPUT_VIDEO if args.save else None,
)

logger.save(f"{config.LOG_DIR}/run_{args.mode}.csv")
print(logger.summary())