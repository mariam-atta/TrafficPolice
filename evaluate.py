import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from preprocess import load_data, split_features_labels


RESULTS_DIR = "results"
MODELS_DIR = "models"

os.makedirs(RESULTS_DIR, exist_ok=True)


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def plot_metric(df, metric, filename):
    plt.figure(figsize=(9, 5))
    plt.bar(df["model"], df[metric])

    plt.title(f"{metric.replace('_', ' ').title()} Comparison")
    plt.xlabel("Model")
    plt.ylabel(metric.replace("_", " ").title())
    plt.ylim(0, 1)

    plt.xticks(rotation=20)
    plt.tight_layout()

    path = f"{RESULTS_DIR}/{filename}"
    plt.savefig(path)
    plt.close()

    print(f"Saved: {path}")


def save_best_model_confusion_matrix():
    print_header("STEP 5: CONFUSION MATRIX FOR BEST MODEL")

    train_df, test_df = load_data()
    X_test, y_test = split_features_labels(test_df)

    model = joblib.load(f"{MODELS_DIR}/best_model.pkl")

    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Attack"]
    )

    display.plot(values_format="d")
    plt.title("Confusion Matrix - Best Model")
    plt.tight_layout()

    path = f"{RESULTS_DIR}/confusion_matrix_best_model.png"
    plt.savefig(path)
    plt.close()

    print(f"Saved: {path}")


def main():
    print_header("STEP 1: LOADING MODEL RESULTS")

    supervised_path = f"{RESULTS_DIR}/supervised_results.csv"
    unsupervised_path = f"{RESULTS_DIR}/unsupervised_results.csv"

    supervised_df = pd.read_csv(supervised_path)
    unsupervised_df = pd.read_csv(unsupervised_path)

    supervised_df["type"] = "Supervised"
    unsupervised_df["type"] = "Unsupervised"

    final_df = pd.concat([supervised_df, unsupervised_df], ignore_index=True)

    print("\nFinal Model Comparison:")
    print(final_df.to_string(index=False))

    print_header("STEP 2: SAVING FINAL COMPARISON CSV")

    final_path = f"{RESULTS_DIR}/final_model_comparison.csv"
    final_df.to_csv(final_path, index=False)
    print(f"Saved: {final_path}")

    print_header("STEP 3: CREATING GRAPHS")

    plot_metric(final_df, "accuracy", "accuracy_comparison.png")
    plot_metric(final_df, "precision", "precision_comparison.png")
    plot_metric(final_df, "recall", "recall_comparison.png")
    plot_metric(final_df, "f1_score", "f1_score_comparison.png")

    print_header("STEP 4: BEST MODEL SELECTION")

    best_model = final_df.sort_values(by="f1_score", ascending=False).iloc[0]

    print(f"Best Model : {best_model['model']}")
    print(f"Model Type : {best_model['type']}")
    print(f"F1 Score   : {best_model['f1_score']:.4f}")
    print(f"Accuracy   : {best_model['accuracy']:.4f}")
    print(f"Precision  : {best_model['precision']:.4f}")
    print(f"Recall     : {best_model['recall']:.4f}")

    save_best_model_confusion_matrix()

    print_header("EVALUATION COMPLETE")


if __name__ == "__main__":
    main()