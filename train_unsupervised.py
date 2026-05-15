import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess import load_data, split_features_labels, TargetEncoder


MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def convert_unsupervised_predictions(predictions):
    """
    Isolation Forest and One-Class SVM return:
    1  = normal
    -1 = anomaly

    Our dataset label is:
    0 = normal
    1 = attack

    So we convert:
    1  -> 0
    -1 -> 1
    """
    return [0 if p == 1 else 1 for p in predictions]


def evaluate_model(model_name, model, X_test, y_test):
    raw_predictions = model.predict(X_test)
    predictions = convert_unsupervised_predictions(raw_predictions)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print_header(f"{model_name} RESULTS")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def main():
    print_header("STEP 1: LOADING DATA")

    train_df, test_df = load_data()

    X_train, y_train = split_features_labels(train_df)
    X_test, y_test = split_features_labels(test_df)

    print(f"Training Rows : {X_train.shape[0]}")
    print(f"Testing Rows  : {X_test.shape[0]}")
    print(f"Features      : {X_train.shape[1]}")

    categorical_cols = ["proto", "service", "state"]

    results = []

    # For unsupervised learning, train mostly on normal traffic
    X_train_normal = X_train[y_train == 0]

    print("\nNormal samples used for unsupervised training:", X_train_normal.shape[0])

    # --------------------------------------------------
    # Model 1: Isolation Forest
    # --------------------------------------------------
    print_header("STEP 2: TRAINING ISOLATION FOREST")

    isolation_pipeline = Pipeline([
        ("target_encoder", TargetEncoder(columns=categorical_cols, smoothing=10)),
        ("model", IsolationForest(
            n_estimators=100,
            contamination=0.3,
            random_state=42,
            n_jobs=-1
        ))
    ])

    isolation_pipeline.fit(X_train_normal, [0] * len(X_train_normal))

    joblib.dump(isolation_pipeline, f"{MODEL_DIR}/isolation_forest_pipeline.pkl")
    print(f"Saved: {MODEL_DIR}/isolation_forest_pipeline.pkl")

    isolation_result = evaluate_model(
        "Isolation Forest",
        isolation_pipeline,
        X_test,
        y_test
    )

    results.append(isolation_result)

    # --------------------------------------------------
    # Model 2: One-Class SVM
    # --------------------------------------------------
    print_header("STEP 3: TRAINING ONE-CLASS SVM")

    # One-Class SVM can be slow, so we use a smaller normal sample
    X_svm_train = X_train_normal.sample(n=min(10000, len(X_train_normal)), random_state=42)

    oneclass_pipeline = Pipeline([
        ("target_encoder", TargetEncoder(columns=categorical_cols, smoothing=10)),
        ("scaler", StandardScaler()),
        ("model", OneClassSVM(
            kernel="rbf",
            nu=0.3,
            gamma="scale"
        ))
    ])

    oneclass_pipeline.fit(X_svm_train, [0] * len(X_svm_train))

    joblib.dump(oneclass_pipeline, f"{MODEL_DIR}/oneclass_svm_pipeline.pkl")
    print(f"Saved: {MODEL_DIR}/oneclass_svm_pipeline.pkl")

    oneclass_result = evaluate_model(
        "One-Class SVM",
        oneclass_pipeline,
        X_test,
        y_test
    )

    results.append(oneclass_result)

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    print_header("STEP 4: SAVING UNSUPERVISED RESULTS")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{RESULTS_DIR}/unsupervised_results.csv", index=False)

    print("\nUnsupervised Model Comparison:")
    print(results_df.to_string(index=False))

    print(f"\nSaved: {RESULTS_DIR}/unsupervised_results.csv")
    print_header("UNSUPERVISED TRAINING COMPLETE")


if __name__ == "__main__":
    main()