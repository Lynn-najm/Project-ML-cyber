import os
import shutil
from datetime import datetime

from utils import fail, save_json


# =========================================================
# SETTINGS
# =========================================================
RUNS_DIR = os.path.join("artifacts", "runs")
FINAL_DIR = os.path.join("artifacts", "final")

# Current winner prefix
MODEL_PREFIX = "random_forest"

# Files we want to promote
REQUIRED_FILES = {
    "model.joblib": "best_model.joblib",
    "metrics.json": "best_model_metrics.json",
    "feature_order.json": "feature_order.json",
    "label_mapping.json": "label_mapping.json",
}

# Optional extra files to keep for convenience
OPTIONAL_FILES = {
    "classification_report.txt": "classification_report.txt",
    "confusion_matrix.csv": "confusion_matrix.csv",
    "feature_importances.csv": "feature_importances.csv",
}


# =========================================================
# HELPERS
# =========================================================
def find_latest_run(runs_dir: str, model_prefix: str) -> str:
    if not os.path.exists(runs_dir):
        fail(f"Runs directory not found: {runs_dir}")

    candidates = []
    for name in os.listdir(runs_dir):
        full_path = os.path.join(runs_dir, name)

        if not os.path.isdir(full_path) or not name.startswith(model_prefix + "_"):
            continue

        # Expected format: random_forest_YYYYMMDD_HHMMSS
        timestamp_str = name[len(model_prefix) + 1:]
        try:
            run_dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            candidates.append((run_dt, full_path))
        except ValueError:
            # Skip folders that don't match the expected naming format
            continue

    if not candidates:
        fail(f"No valid run folders found for prefix '{model_prefix}' inside {runs_dir}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def copy_required_files(source_dir: str, final_dir: str) -> None:
    for src_name, dst_name in REQUIRED_FILES.items():
        src_path = os.path.join(source_dir, src_name)
        dst_path = os.path.join(final_dir, dst_name)

        if not os.path.exists(src_path):
            fail(f"Required file missing in selected run: {src_path}")

        shutil.copy2(src_path, dst_path)


def copy_optional_files(source_dir: str, final_dir: str) -> list[str]:
    copied = []
    for src_name, dst_name in OPTIONAL_FILES.items():
        src_path = os.path.join(source_dir, src_name)
        dst_path = os.path.join(final_dir, dst_name)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied.append(dst_name)

    return copied


# =========================================================
# 1) FIND LATEST WINNING RUN
# =========================================================
source_run_dir = find_latest_run(RUNS_DIR, MODEL_PREFIX)

print("=" * 60)
print("🏁 PROMOTE CURRENT BEST MODEL")
print("=" * 60)
print(f"Selected source run: {source_run_dir}")

# =========================================================
# 2) CREATE / REFRESH FINAL DIRECTORY
# =========================================================
if os.path.exists(FINAL_DIR):
    shutil.rmtree(FINAL_DIR)

os.makedirs(FINAL_DIR)

# =========================================================
# 3) COPY FILES
# =========================================================
copy_required_files(source_run_dir, FINAL_DIR)
copied_optional = copy_optional_files(source_run_dir, FINAL_DIR)

# =========================================================
# 4) SAVE PROMOTION SUMMARY
# =========================================================
summary = {
    "promoted_model_prefix": MODEL_PREFIX,
    "source_run_dir": source_run_dir,
    "final_dir": FINAL_DIR,
    "required_files_promoted": list(REQUIRED_FILES.values()),
    "optional_files_promoted": copied_optional,
    "promoted_at": datetime.now().isoformat(timespec="seconds"),
}

summary_path = os.path.join(FINAL_DIR, "promotion_summary.json")
save_json(summary, summary_path)

# =========================================================
# 5) DONE
# =========================================================
print("\nPromoted files:")
for dst_name in REQUIRED_FILES.values():
    print(f"  - {os.path.join(FINAL_DIR, dst_name)}")

for dst_name in copied_optional:
    print(f"  - {os.path.join(FINAL_DIR, dst_name)}")

print(f"  - {summary_path}")

print("\n✅ Best model promotion complete.")