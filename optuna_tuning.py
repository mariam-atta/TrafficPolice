import os
import joblib
import optuna
import pandas as pd

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess import load_data, split_features_labels, TargetEncoder


# Folders
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def calculate_scores(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }


def main():
    print_header("STEP 1: LOADING DATA")

    train_df, test_df = load_data()

    X_train_full, y_train_full = split_features_labels(train_df)
    X_test, y_test = split_features_labels(test_df)

    print("Full Training Rows:", X_train_full.shape[0])
    print("Testing Rows      :", X_test.shape[0])
    print("Features          :", X_train_full.shape[1])

    # Categorical columns for target encoding
    categorical_cols = ["proto", "service", "state"]

    print_header("STEP 2: TRAIN / VALIDATION SPLIT")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )

    print("Optuna Train Rows     :", X_train.shape[0])
    print("Optuna Validation Rows:", X_valid.shape[0])

    print_header("STEP 3: OPTUNA HYPERPARAMETER TUNING")

    def objective(trial):
        # Optuna chooses these values
        n_estimators = trial.suggest_int("n_estimators", 50, 150)
        max_depth = trial.suggest_int("max_depth", 10, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

        # Same pipeline style as our supervised training
        model_pipeline = Pipeline([
            ("target_encoder", TargetEncoder(columns=categorical_cols, smoothing=10)),
            ("smote", SMOTE(random_state=42)),
            ("model", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1
            ))
        ])

        model_pipeline.fit(X_train, y_train)

        valid_predictions = model_pipeline.predict(X_valid)
        valid_f1 = f1_score(y_valid, valid_predictions, zero_division=0)

        return valid_f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("\nBest Validation F1:", study.best_value)
    print("Best Parameters:")
    print(study.best_params)

    print_header("STEP 4: TRAINING FINAL OPTUNA MODEL")

    best_params = study.best_params

    final_pipeline = Pipeline([
        ("target_encoder", TargetEncoder(columns=categorical_cols, smoothing=10)),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Train final tuned model on full training data
    final_pipeline.fit(X_train_full, y_train_full)

    print_header("STEP 5: TESTING OPTUNA MODEL")

    test_predictions = final_pipeline.predict(X_test)
    scores = calculate_scores(y_test, test_predictions)

    result = {
        "model": "Random Forest + SMOTE + Optuna",
        "accuracy": scores["accuracy"],
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1_score": scores["f1_score"],
        "best_params": str(best_params)
    }

    results_df = pd.DataFrame([result])

    print("\nOptuna Test Results:")
    print(results_df.to_string(index=False))

    # Save results and tuned model
    results_df.to_csv("results/optuna_results.csv", index=False)
    joblib.dump(final_pipeline, "models/random_forest_optuna_pipeline.pkl")

    with open("results/optuna_best_params.txt", "w") as file:
        file.write("Optuna HPO Results\n")
        file.write("==================\n")
        file.write(f"Best Validation F1: {study.best_value}\n")
        file.write(f"Best Parameters: {best_params}\n")
        file.write("\nFinal Test Results:\n")
        file.write(results_df.to_string(index=False))

    print("\nSaved:")
    print("results/optuna_results.csv")
    print("results/optuna_best_params.txt")
    print("models/random_forest_optuna_pipeline.pkl")

    print_header("OPTUNA TUNING COMPLETE")


if __name__ == "__main__":
    main()