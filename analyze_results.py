import matplotlib.pyplot as plt
import pandas as pd

from src import config

df = pd.read_csv(f"{config.LOG_DIR}/experiments.csv")

def plot(exp, xlabel, fname):
    d = df[df["experiment"] == exp]
    plt.figure(figsize=(6, 4))
    plt.bar(d["setting"].astype(str), d["mean_fps"])
    plt.xlabel(xlabel)
    plt.ylabel("Mean FPS")
    plt.title(f"FPS vs {xlabel}")
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/{fname}", dpi=150)
    plt.close()

plot("kernel", "Gaussian kernel size", "fps_kernel.png")
plot("model", "YOLO model", "fps_model.png")
plot("resolution", "Frame width (px)", "fps_resolution.png")
plot("mode", "Pipeline mode", "fps_mode.png")

# Per-stage latency breakdown from a full run
run = pd.read_csv(f"{config.LOG_DIR}/run_both.csv")
stages = [c for c in run.columns if c.endswith("_ms") and c != "total_ms"]
run[stages].mean().plot(kind="bar", figsize=(7, 4), title="Mean latency per stage (ms)")
plt.ylabel("ms")
plt.tight_layout()
plt.savefig(f"{config.PLOT_DIR}/stage_breakdown.png", dpi=150)
print("Plots saved.")