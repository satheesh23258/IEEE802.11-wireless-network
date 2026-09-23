import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from src.preprocessing import build_wireless_features, load_wireless_training_data
except ModuleNotFoundError:
    from preprocessing import build_wireless_features, load_wireless_training_data


def run_training_pipeline(data_path):
    paths = data_path if isinstance(data_path, (list, tuple)) else [data_path]
    dataset = load_wireless_training_data(paths)
    X, y = build_wireless_features(dataset)
    if X.empty or y.empty:
        raise ValueError("No valid training rows were found in the dataset.")

    labels = sorted(y.unique().tolist())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    candidates = {
        "random_forest": RandomForestClassifier(random_state=42, n_estimators=200, n_jobs=-1),
        "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=16),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42, n_estimators=200),
    }
    trained = {}
    comparison = []
    for name, classifier in candidates.items():
        candidate = Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])
        candidate.fit(X_train, y_train)
        candidate_predictions = candidate.predict(X_test)
        candidate_report = classification_report(y_test, candidate_predictions, labels=labels, output_dict=True, zero_division=0)
        comparison.append({"model": name, "accuracy": accuracy_score(y_test, candidate_predictions), "macro_f1": candidate_report["macro avg"]["f1-score"]})
        trained[name] = (candidate, candidate_predictions, candidate_report)
    comparison_frame = pd.DataFrame(comparison).sort_values("macro_f1", ascending=False)
    best_name = comparison_frame.iloc[0]["model"]
    pipeline, predictions, report = trained[best_name]
    Path("models").mkdir(exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, "models/best_model.pkl")
    pd.DataFrame(report).transpose().to_csv("results/metrics/classification_report.csv")
    comparison_frame.to_csv("results/metrics/model_comparison.csv", index=False)
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv("results/metrics/confusion_matrix.csv")
    classifier = pipeline.named_steps["classifier"]
    importances = getattr(classifier, "feature_importances_", [0.0] * len(X.columns))
    metadata = {
        "model": best_name,
        "features": list(X.columns),
        "target_classes": labels,
        "test_accuracy": accuracy_score(y_test, predictions),
        "macro_f1": report["macro avg"]["f1-score"],
        "feature_importance": dict(zip(X.columns, importances)),
    }
    Path("models/model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Selected model: {best_name}")
    print("Training complete. Saved models/best_model.pkl")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Macro F1-score: {report['macro avg']['f1-score']:.4f}")


if __name__ == "__main__":
    default_datasets = [
        "data/WiFi_Transmission_Rate_Recommendation_Dataset_5000.xlsx",
        "data/dataset_complete.csv",
        "data/dataset_P_A_auto.csv",
    ]
    existing = [path for path in default_datasets if Path(path).exists()]
    run_training_pipeline(existing if existing else default_datasets)
