import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


TRAIN_PATH = "data/UNSW_NB15_training-set.csv"
TEST_PATH = "data/UNSW_NB15_testing-set.csv"


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def split_features_labels(df):
    df = df.copy()

    df = df.drop(columns=["id", "attack_cat"], errors="ignore")

    X = df.drop(columns=["label"])
    y = df["label"].astype(int)

    return X, y


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, smoothing=10):
        self.columns = columns
        self.smoothing = smoothing
        self.global_mean = None
        self.encoders = {}

    def fit(self, X, y):
        X = X.copy()
        y = pd.Series(y)

        self.global_mean = y.mean()

        for col in self.columns:
            stats = y.groupby(X[col]).agg(["mean", "count"])
            smooth = (
                stats["count"] * stats["mean"] +
                self.smoothing * self.global_mean
            ) / (stats["count"] + self.smoothing)

            self.encoders[col] = smooth.to_dict()

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.columns:
            X[col] = X[col].map(self.encoders[col]).fillna(self.global_mean)

        return X


if __name__ == "__main__":
    train_df, test_df = load_data()

    X_train, y_train = split_features_labels(train_df)
    X_test, y_test = split_features_labels(test_df)

    print("Train:", train_df.shape)
    print("Test:", test_df.shape)
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:")
    print(y_train.value_counts())
    print("y_test:")
    print(y_test.value_counts())