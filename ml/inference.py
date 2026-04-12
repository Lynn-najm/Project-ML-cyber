import json
import os
from functools import lru_cache

import joblib
import pandas as pd

try:
    from .utils import fail
except ImportError:
    from utils import fail


FINAL_DIR = os.path.join("artifacts", "final")
MODEL_PATH = os.path.join(FINAL_DIR, "best_model.joblib")
FEATURE_ORDER_PATH = os.path.join(FINAL_DIR, "feature_order.json")
LABEL_MAPPING_PATH = os.path.join(FINAL_DIR, "label_mapping.json")


@lru_cache(maxsize=1)
def load_final_artifacts():
    """
    Load and cache the final promoted model and its metadata.
    This avoids reloading files on every prediction call.
    """
    if not os.path.exists(MODEL_PATH):
        fail(f"Final model not found: {MODEL_PATH}")

    if not os.path.exists(FEATURE_ORDER_PATH):
        fail(f"Feature order file not found: {FEATURE_ORDER_PATH}")

    if not os.path.exists(LABEL_MAPPING_PATH):
        fail(f"Label mapping file not found: {LABEL_MAPPING_PATH}")

    model = joblib.load(MODEL_PATH)

    with open(FEATURE_ORDER_PATH, "r") as f:
        feature_order = json.load(f)

    with open(LABEL_MAPPING_PATH, "r") as f:
        label_mapping = json.load(f)

    saved_labels = label_mapping.get("classes", [])

    return model, feature_order, saved_labels


def validate_input(features_dict: dict, feature_order: list) -> tuple[list, list]:
    """
    Check missing and extra fields.
    Returns:
        missing_features, extra_features
    """
    incoming_keys = set(features_dict.keys())
    required_keys = set(feature_order)

    missing_features = [f for f in feature_order if f not in incoming_keys]
    extra_features = sorted(list(incoming_keys - required_keys))

    return missing_features, extra_features


def build_input_dataframe(features_dict: dict, feature_order: list) -> pd.DataFrame:
    """
    Reorder input fields exactly as expected by the model.
    Extra fields are ignored.
    """
    row = {feature: features_dict[feature] for feature in feature_order}
    return pd.DataFrame([row], columns=feature_order)


def predict(features_dict: dict) -> dict:
    """
    Predict one sample from a dictionary of features.

    Example input:
    {
        "Header_Length": 123.0,
        "Protocol Type": 6,
        ...
    }

    Example output:
    {
        "prediction": "ATTACK",
        "confidence": 0.87,
        "probabilities": {
            "ATTACK": 0.87,
            "BENIGN": 0.13
        },
        "missing_features": [],
        "extra_features_ignored": []
    }
    """
    if not isinstance(features_dict, dict):
        fail("Input to predict() must be a dictionary of feature_name -> value.")

    model, feature_order, saved_labels = load_final_artifacts()

    missing_features, extra_features = validate_input(features_dict, feature_order)

    if missing_features:
        fail(f"Missing required features: {missing_features}")

    X_input = build_input_dataframe(features_dict, feature_order)

    pred = model.predict(X_input)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_input)[0]

        final_estimator = model
        if hasattr(model, "named_steps") and len(model.named_steps) > 0:
            final_step_name = list(model.named_steps.keys())[-1]
            final_estimator = model.named_steps[final_step_name]

        if hasattr(final_estimator, "classes_"):
            class_order = list(final_estimator.classes_)
        else:
            class_order = saved_labels if saved_labels else []

        prob_dict = {
            str(label): float(prob)
            for label, prob in zip(class_order, probabilities)
        }

        confidence = max(prob_dict.values()) if prob_dict else None
    else:
        prob_dict = {}
        confidence = None

    return {
        "prediction": str(pred),
        "confidence": float(confidence) if confidence is not None else None,
        "probabilities": prob_dict,
        "missing_features": missing_features,
        "extra_features_ignored": extra_features,
    }


if __name__ == "__main__":
    # Small manual test example
    model, feature_order, _ = load_final_artifacts()

    dummy_input = {feature: 0 for feature in feature_order}

    result = predict(dummy_input)
    print(json.dumps(result, indent=4))