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
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from project_utils import load_cleaned_data, take_stratified_sample, train_test_split


def compare_booking_models(
    cleaned_data,
    target_column="is_canceled",
    sample_size=20_000,
    test_size=0.2,
    cv_splits=5,
    random_state=42,
):
    """Compare preprocessing and classification model combinations.

    The function samples a cleaned booking dataset, creates a stratified
    training/test split, evaluates preprocessing and model combinations using
    stratified k-fold cross-validation on the training set, and evaluates the
    selected best model on the untouched test set.

    Parameters
    ----------
    cleaned_data : pandas.DataFrame
        Cleaned booking dataset containing predictor columns and the binary
        target column. Outcome leakage columns must already be removed.
    target_column : str, default="is_canceled"
        Name of the classification target column.
    sample_size : int or None, default=20000
        Number of stratified records used for comparison. If ``None`` or
        greater than the dataset size, all available records are used.
    test_size : float, default=0.2
        Fraction of sampled records reserved for final test evaluation.
    cv_splits : int, default=5
        Number of stratified cross-validation folds applied to training data.
    random_state : int, default=42
        Random seed used for sampling, splitting, and tree models.

    Returns
    -------
    dict
        Results containing all cross-validation combinations, the five
        highest-ranked combinations, the selected combination, final test
        scores, and the fitted best pipeline.

    Examples
    --------
    >>> output = compare_booking_models(load_cleaned_data())
    >>> output["top_five"][["model", "f1_score"]]
    >>> output["test_scores"]["roc_auc"]
    """
    if target_column not in cleaned_data.columns:
        raise ValueError(f"Target column '{target_column}' is not in cleaned_data.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be a float between 0 and 1.")
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2.")

    data = take_stratified_sample(cleaned_data, sample_size, random_state)
    X = data.drop(columns=target_column)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    categorical_columns = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()
    numerical_columns = X_train.select_dtypes(include=["number"]).columns.tolist()

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

    def make_pipeline(scaler, encoder, model):
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

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    result_rows = []
    for scaler_name, scaler in scalers.items():
        for encoder_name, encoder in encoders.items():
            for model_name, parameters, model in models:
                fold_rows = []
                for train_index, validation_index in cv.split(X_train, y_train):
                    pipeline = make_pipeline(scaler, encoder, model)
                    pipeline.fit(X_train.iloc[train_index], y_train.iloc[train_index])
                    prediction = pipeline.predict(X_train.iloc[validation_index])
                    actual = y_train.iloc[validation_index]
                    fold_rows.append(
                        {
                            "accuracy": accuracy_score(actual, prediction),
                            "balanced_accuracy": balanced_accuracy_score(
                                actual, prediction
                            ),
                            "precision": precision_score(
                                actual, prediction, zero_division=0
                            ),
                            "recall": recall_score(actual, prediction, zero_division=0),
                            "f1_score": f1_score(actual, prediction, zero_division=0),
                        }
                    )

                mean_scores = pd.DataFrame(fold_rows).mean()
                result_rows.append(
                    {
                        "scaling": scaler_name,
                        "encoding": encoder_name,
                        "model": model_name,
                        "parameters": parameters,
                        **mean_scores.to_dict(),
                    }
                )

    all_results = (
        pd.DataFrame(result_rows)
        .sort_values(["f1_score", "balanced_accuracy"], ascending=False)
        .reset_index(drop=True)
    )
    top_five = all_results.head(5).copy()
    selected_combination = all_results.iloc[0].copy()
    selected_model = next(
        model
        for model_name, parameters, model in models
        if model_name == selected_combination["model"]
        and parameters == selected_combination["parameters"]
    )
    best_model = make_pipeline(
        scalers[selected_combination["scaling"]],
        encoders[selected_combination["encoding"]],
        selected_model,
    )
    best_model.fit(X_train, y_train)
    test_prediction = best_model.predict(X_test)
    test_probability = best_model.predict_proba(X_test)[:, 1]
    test_scores = pd.Series(
        {
            "accuracy": accuracy_score(y_test, test_prediction),
            "balanced_accuracy": balanced_accuracy_score(y_test, test_prediction),
            "precision": precision_score(y_test, test_prediction, zero_division=0),
            "recall": recall_score(y_test, test_prediction, zero_division=0),
            "f1_score": f1_score(y_test, test_prediction, zero_division=0),
            "roc_auc": roc_auc_score(y_test, test_probability),
        }
    )

    return {
        "all_results": all_results,
        "top_five": top_five,
        "selected_combination": selected_combination,
        "test_scores": test_scores,
        "best_model": best_model,
    }


output = compare_booking_models(load_cleaned_data())
selected = output["selected_combination"]

print("Top five combinations based on training CV F1-score")
print(output["top_five"].round(4).to_string(index=False))
print("\nFinal selected combination")
print(selected[["scaling", "encoding", "model", "parameters"]].to_string())
print("\nFinal test set results")
print(output["test_scores"].round(4).to_string())
