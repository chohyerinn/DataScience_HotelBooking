import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ProjectUtils import load_cleaned_data, take_stratified_sample


RANDOM_STATE = 42


def get_models(random_state=RANDOM_STATE):
    return [
        (
            "Logistic Regression",
            "C=0.1, class_weight=balanced",
            LogisticRegression(
                C=0.1, class_weight="balanced", solver="liblinear", max_iter=2000
            ),
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
            "max_depth=3, class_weight=balanced",
            DecisionTreeClassifier(
                max_depth=3, class_weight="balanced", random_state=random_state
            ),
        ),
        (
            "Decision Tree",
            "max_depth=5, class_weight=balanced",
            DecisionTreeClassifier(
                max_depth=5, class_weight="balanced", random_state=random_state
            ),
        ),
        (
            "Decision Tree",
            "max_depth=8, class_weight=balanced",
            DecisionTreeClassifier(
                max_depth=8, class_weight="balanced", random_state=random_state
            ),
        ),
        ("K-Nearest Neighbors", "n_neighbors=5", KNeighborsClassifier(n_neighbors=5)),
        (
            "K-Nearest Neighbors",
            "n_neighbors=11",
            KNeighborsClassifier(n_neighbors=11),
        ),
        (
            "K-Nearest Neighbors",
            "n_neighbors=21",
            KNeighborsClassifier(n_neighbors=21),
        ),
    ]


def make_pipeline(X, scaler, encoder, model):
    categorical_columns = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", clone(scaler)),
                    ]
                ),
                numerical_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", clone(encoder)),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", clone(model))])


def compare_booking_models(X, y, cv_splits=5, random_state=RANDOM_STATE):
    """Compare preprocessing and model combinations using training data only."""
    scalers = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler()}
    encoders = {
        "OneHotEncoder": OneHotEncoder(handle_unknown="ignore"),
        "OrdinalEncoder": OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ),
    }
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    results = []

    for scaler_name, scaler in scalers.items():
        for encoder_name, encoder in encoders.items():
            for model_name, parameters, model in get_models(random_state):
                fold_results = []
                for train_index, validation_index in cv.split(X, y):
                    pipeline = make_pipeline(X, scaler, encoder, model)
                    pipeline.fit(X.iloc[train_index], y.iloc[train_index])
                    prediction = pipeline.predict(X.iloc[validation_index])
                    actual = y.iloc[validation_index]
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


def evaluate_selected_model(X_train, y_train, X_test, y_test, selected):
    scalers = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler()}
    encoders = {
        "OneHotEncoder": OneHotEncoder(handle_unknown="ignore"),
        "OrdinalEncoder": OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ),
    }
    model_lookup = {
        (model_name, parameters): model
        for model_name, parameters, model in get_models(RANDOM_STATE)
    }
    pipeline = make_pipeline(
        X_train,
        scalers[selected["scaling"]],
        encoders[selected["encoding"]],
        model_lookup[(selected["model"], selected["parameters"])],
    )
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)
    probability = pipeline.predict_proba(X_test)[:, 1]
    return pd.Series(
        {
            "accuracy": accuracy_score(y_test, prediction),
            "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
            "precision": precision_score(y_test, prediction, zero_division=0),
            "recall": recall_score(y_test, prediction, zero_division=0),
            "f1_score": f1_score(y_test, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probability),
        }
    )


if __name__ == "__main__":
    df = take_stratified_sample(load_cleaned_data(), 20_000, RANDOM_STATE)
    X = df.drop(columns="is_canceled")
    y = df["is_canceled"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results = compare_booking_models(X_train, y_train)
    selected = results.iloc[0]
    test_results = evaluate_selected_model(
        X_train, y_train, X_test, y_test, selected
    )

    print("Top five combinations based on training CV F1-score")
    print(results.head(5).round(4).to_string(index=False))
    print("\nFinal selected combination")
    print(selected[["scaling", "encoding", "model", "parameters"]].to_string())
    print("\nFinal test set results")
    print(test_results.round(4).to_string())
