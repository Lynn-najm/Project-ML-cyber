import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from utils import (
    load_config,
    load_dataset,
    get_features_and_target,
    validate_schema,
    replace_infinities_with_nan,
    compute_benign_fpr,
    save_json,
    make_run_dir,
    make_stratified_split,
)


# =========================================================
# SETTINGS
# =========================================================
SEEDS = [7, 21, 42, 84, 123]
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Analysis-only outputs
ANALYSIS_DIR = make_run_dir(
    "repeated_seed_eval",
    base_dir=os.path.join("artifacts", "analysis")
)


# =========================================================
# HELPERS
# =========================================================
def evaluate_model(model_name, pipeline, X_train, y_train, X_val, y_val):
    """
    Fit the model and compute train/validation metrics.
    """
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_val_pred = pipeline.predict(X_val)

    labels = None
    if hasattr(pipeline, "named_steps"):
        final_step_name = list(pipeline.named_steps.keys())[-1]
        final_estimator = pipeline.named_steps[final_step_name]
        if hasattr(final_estimator, "classes_"):
            labels = list(final_estimator.classes_)

    if labels is None:
        labels = sorted(y_val.unique().tolist())

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_macro_f1 = f1_score(y_train, y_train_pred, average="macro")

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_macro_f1 = f1_score(y_val, y_val_pred, average="macro")

    cm = confusion_matrix(y_val, y_val_pred, labels=labels)
    fpr_benign, _ = compute_benign_fpr(cm, labels)

    return {
        "model": model_name,
        "train_accuracy": float(train_accuracy),
        "train_macro_f1": float(train_macro_f1),
        "validation_accuracy": float(val_accuracy),
        "validation_macro_f1": float(val_macro_f1),
        "false_positive_rate_benign": float(fpr_benign) if fpr_benign is not None else None,
        "class_order": labels,
    }


# =========================================================
# 1) LOAD DATA
# =========================================================
df = load_dataset("clean_sample.csv")
config = load_config("feature_list.json")

features, target = get_features_and_target(config)
validate_schema(df, features, target)

X = df[features].copy()
y = df[target].copy()

print("=" * 60)
print("🔁 REPEATED-SEED EVALUATION")
print("=" * 60)
print(f"Dataset shape: {df.shape}")
print(f"Target column: {target}")
print(f"Number of features: {len(features)}")
print(f"Seeds: {SEEDS}")

print("\nOverall class distribution:")
print(y.value_counts())

# Clean infinities once
X, inf_counts = replace_infinities_with_nan(X)

if not inf_counts.empty:
    print("\n⚠️ Infinite values detected and replaced with NaN:")
    print(inf_counts)
else:
    print("\n✅ No infinite values detected.")


# =========================================================
# 2) LOOP OVER SEEDS
# =========================================================
all_results = []

for seed in SEEDS:
    print("\n" + "-" * 60)
    print(f"Running seed: {seed}")
    print("-" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test = make_stratified_split(
        X,
        y,
        seed=seed,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
    )

    print(f"Train shape      : {X_train.shape}")
    print(f"Validation shape : {X_val.shape}")
    print(f"Test shape       : {X_test.shape}")

    logistic_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])

    rf_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ))
    ])

    logistic_result = evaluate_model(
        "LogisticRegression",
        logistic_pipeline,
        X_train, y_train, X_val, y_val
    )
    logistic_result["seed"] = seed
    all_results.append(logistic_result)

    rf_result = evaluate_model(
        "RandomForestClassifier",
        rf_pipeline,
        X_train, y_train, X_val, y_val
    )
    rf_result["seed"] = seed
    all_results.append(rf_result)

    print("Logistic Regression:")
    print(f"  Train Acc  : {logistic_result['train_accuracy']:.6f}")
    print(f"  Val Acc    : {logistic_result['validation_accuracy']:.6f}")
    print(f"  Train F1   : {logistic_result['train_macro_f1']:.6f}")
    print(f"  Val F1     : {logistic_result['validation_macro_f1']:.6f}")
    fpr_val= logistic_result['false_positive_rate_benign']
    print(f"  Benign FPR : {fpr_val:.6f}" if fpr_val is not None else " Benign FPR : N/A")

    print("Random Forest:")
    print(f"  Train Acc  : {rf_result['train_accuracy']:.6f}")
    print(f"  Val Acc    : {rf_result['validation_accuracy']:.6f}")
    print(f"  Train F1   : {rf_result['train_macro_f1']:.6f}")
    print(f"  Val F1     : {rf_result['validation_macro_f1']:.6f}")
    fpr_val= rf_result['false_positive_rate_benign']
    print(f"  Benign FPR : {fpr_val:.6f}" if fpr_val is not None else " Benign FPR :N/A")


# =========================================================
# 3) SAVE PER-SEED RESULTS
# =========================================================
results_df = pd.DataFrame(all_results)
results_csv_path = os.path.join(ANALYSIS_DIR, "per_seed_results.csv")
results_df.to_csv(results_csv_path, index=False)

print("\n" + "=" * 60)
print("📊 PER-SEED RESULTS")
print("=" * 60)
print(results_df[[
    "seed",
    "model",
    "train_accuracy",
    "validation_accuracy",
    "train_macro_f1",
    "validation_macro_f1",
    "false_positive_rate_benign"
]].to_string(index=False))


# =========================================================
# 4) SUMMARY STATISTICS
# =========================================================
summary_df = (
    results_df.groupby("model")[[
        "train_accuracy",
        "validation_accuracy",
        "train_macro_f1",
        "validation_macro_f1",
        "false_positive_rate_benign"
    ]]
    .agg(["mean", "std", "min", "max"])
)

summary_csv_path = os.path.join(ANALYSIS_DIR, "summary_statistics.csv")
summary_df.to_csv(summary_csv_path)

print("\n" + "=" * 60)
print("📈 SUMMARY STATISTICS")
print("=" * 60)
print(summary_df)

summary_json = {}

for model_name in results_df["model"].unique():
    model_rows = results_df[results_df["model"] == model_name]

    summary_json[model_name] = {
        "num_seeds": int(len(model_rows)),
        "train_accuracy_mean": float(model_rows["train_accuracy"].mean()),
        "train_accuracy_std": float(model_rows["train_accuracy"].std(ddof=1)),
        "validation_accuracy_mean": float(model_rows["validation_accuracy"].mean()),
        "validation_accuracy_std": float(model_rows["validation_accuracy"].std(ddof=1)),
        "train_macro_f1_mean": float(model_rows["train_macro_f1"].mean()),
        "train_macro_f1_std": float(model_rows["train_macro_f1"].std(ddof=1)),
        "validation_macro_f1_mean": float(model_rows["validation_macro_f1"].mean()),
        "validation_macro_f1_std": float(model_rows["validation_macro_f1"].std(ddof=1)),
        "false_positive_rate_benign_mean": float(model_rows["false_positive_rate_benign"].mean()),
        "false_positive_rate_benign_std": float(model_rows["false_positive_rate_benign"].std(ddof=1)),
    }

summary_json["seeds_used"] = SEEDS
summary_json["notes"] = [
    "This is an analysis-only script.",
    "It does not overwrite the official frozen split or final artifacts.",
    "It is meant to test metric stability across multiple random seeds."
]

summary_json_path = os.path.join(ANALYSIS_DIR, "summary_statistics.json")
save_json(summary_json, summary_json_path)

print("\nSaved analysis artifacts:")
print(f"  - {results_csv_path}")
print(f"  - {summary_csv_path}")
print(f"  - {summary_json_path}")

print(f"\n✅ Repeated-seed evaluation complete: {ANALYSIS_DIR}")