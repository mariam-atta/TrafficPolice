import os
import joblib
import pandas as pd

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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


def print_dataset_info(X_train, y_train, X_test, y_test):
    print_header("DATASET INFORMATION")

    print(f"Training Rows   : {X_train.shape[0]}")
    print(f"Testing Rows    : {X_test.shape[0]}")
    print(f"Total Features  : {X_train.shape[1]}")

    print("\nTraining Label Distribution:")
    print(y_train.value_counts().to_string())

    print("\nTesting Label Distribution:")
    print(y_test.value_counts().to_string())


def evaluate_model(model_name, model, X_test, y_test):
    predictions = model.predict(X_test)

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

    print_dataset_info(X_train, y_train, X_test, y_test)

    categorical_cols = ["proto", "service", "state"]

    results = []

    # --------------------------------------------------
    # Model 1: Logistic Regression + SMOTE
    # --------------------------------------------------
    print_header("STEP 2: TRAINING LOGISTIC REGRESSION WITH SMOTE")

    logistic_pipeline = Pipeline([
        ("target_encoder", TargetEncoder(columns=categorical_cols, smoothing=10)),
        ("smote", SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    logistic_pipeline.fit(X_train, y_train)

    joblib.dump(logistic_pipeline, f"{MODEL_DIR}/logistic_pipeline.pkl")
    print(f"Saved: {MODEL_DIR}/logistic_pipeline.pkl")

    logistic_result = evaluate_model(
        "Logistic Regression + SMOTE",
        logistic_pipeline,
        X_test,
        y_test
    )

    results.append(logistic_result)

    # --------------------------------------------------
    # Model 2: Random Forest + SMOTE
    # --------------------------------------------------
    print_header("STEP 3: TRAINING RANDOM FOREST WITH SMOTE")

    random_forest_pipeline = Pipeline([
        ("target_encoder", TargetEncoder(columns=categorical_cols, smoothing=10)),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ))
    ])

    random_forest_pipeline.fit(X_train, y_train)

    joblib.dump(random_forest_pipeline, f"{MODEL_DIR}/random_forest_pipeline.pkl")
    print(f"Saved: {MODEL_DIR}/random_forest_pipeline.pkl")

    rf_result = evaluate_model(
        "Random Forest + SMOTE",
        random_forest_pipeline,
        X_test,
        y_test
    )

    results.append(rf_result)

    # --------------------------------------------------
    # Save comparison results
    # --------------------------------------------------
    print_header("STEP 4: SAVING RESULTS")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{RESULTS_DIR}/supervised_results.csv", index=False)

    print("\nSupervised Model Comparison:")
    print(results_df.to_string(index=False))

    # Select best model by F1-score
    best_row = results_df.sort_values(by="f1_score", ascending=False).iloc[0]
    best_model_name = best_row["model"]

    if "Random Forest" in best_model_name:
        best_model = random_forest_pipeline
    else:
        best_model = logistic_pipeline

    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")

    print(f"\nBest Model Selected: {best_model_name}")
    print(f"Saved: {MODEL_DIR}/best_model.pkl")
    print(f"Saved: {RESULTS_DIR}/supervised_results.csv")

    print_header("SUPERVISED TRAINING COMPLETE")


if __name__ == "__main__":
    main()