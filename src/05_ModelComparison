import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from project_utils import load_cleaned_data, take_stratified_sample


def compare_booking_models(
    cleaned_df,
    target_column="is_canceled",
    cv_splits=5,
    sample_size=None,
    random_state=42,
):
    """Compare preprocessing and classification model combinations."""
    data = take_stratified_sample(cleaned_df, sample_size, random_state)
    X = data.drop(columns=target_column)
    y = data[target_column]

    categorical_columns = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()

    scalers = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler()}
    encoders = {
        "OneHotEncoder": OneHotEncoder(handle_unknown="ignore"),
        "OrdinalEncoder": OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ),
    }
    models = [
        (
            "Logistic Regression",
            "C=1.0, class_weight=None",
            LogisticRegression(C=1.0, solver="liblinear", max_iter=2000),
        ),
        (
            "Logistic Regression",
            "C=1.0, class_weight=balanced",
            LogisticRegression(
                C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000
            ),
        ),
        (
            "Decision Tree",
            "max_depth=5, class_weight=None",
            DecisionTreeClassifier(max_depth=5, random_state=random_state),
        ),
        (
            "Decision Tree",
            "max_depth=5, class_weight=balanced",
            DecisionTreeClassifier(
                max_depth=5, class_weight="balanced", random_state=random_state
            ),
        ),
        ("K-Nearest Neighbors", "n_neighbors=5", KNeighborsClassifier(n_neighbors=5)),
        (
            "K-Nearest Neighbors",
            "n_neighbors=11",
            KNeighborsClassifier(n_neighbors=11),
        ),
    ]

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    results = []

    for scaler_name, scaler in scalers.items():
        for encoder_name, encoder in encoders.items():
            preprocessor = ColumnTransformer(
                [
                    (
                        "numeric",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", scaler),
                            ]
                        ),
                        numerical_columns,
                    ),
                    (
                        "categorical",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("encoder", encoder),
                            ]
                        ),
                        categorical_columns,
                    ),
                ]
            )

            for model_name, parameters, model in models:
                fold_results = []
                for train_index, test_index in cv.split(X, y):
                    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
                    pipeline.fit(X.iloc[train_index], y.iloc[train_index])
                    prediction = pipeline.predict(X.iloc[test_index])
                    actual = y.iloc[test_index]
                    fold_results.append(
                        {
                            "accuracy": accuracy_score(actual, prediction),
                            "balanced_accuracy": balanced_accuracy_score(actual, prediction),
                            "precision": precision_score(actual, prediction, zero_division=0),
                            "recall": recall_score(actual, prediction, zero_division=0),
                            "f1_score": f1_score(actual, prediction, zero_division=0),
                        }
                    )

                scores = pd.DataFrame(fold_results).mean()
                results.append(
                    {
                        "scaling": scaler_name,
                        "encoding": encoder_name,
                        "model": model_name,
                        "parameters": parameters,
                        **scores.to_dict(),
                    }
                )

    return (
        pd.DataFrame(results)
        .sort_values(["f1_score", "balanced_accuracy"], ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    results = compare_booking_models(load_cleaned_data(), sample_size=10_000)
    print("Top five combinations based on F1-score")
    print(results.head(5).round(4).to_string(index=False))
