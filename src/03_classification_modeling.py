import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from project_utils import load_cleaned_data, take_stratified_sample, train_test_split

RANDOM_STATE = 42
df = take_stratified_sample(load_cleaned_data(), 20_000, RANDOM_STATE)
X = df.drop(columns="is_canceled")
y = df["is_canceled"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

categorical_columns = X.select_dtypes(include=["object", "string"]).columns.tolist()
numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()

# Scaling keeps distance-based KNN and coefficient-based models comparable.
preprocessor = ColumnTransformer(
    [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numerical_columns,
        ),
        (
            "categorical",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_columns,
        ),
    ]
)

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=2000, solver="liblinear"
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=11),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = []
roc_results = {}

for model_name, model in models.items():
    fold_results = []
    actual_values = []
    predicted_values = []
    probability_values = []

    for train_index, test_index in cv.split(X_train, y_train):
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train.iloc[train_index], y_train.iloc[train_index])
        prediction = pipeline.predict(X_train.iloc[test_index])
        probability = pipeline.predict_proba(X_train.iloc[test_index])[:, 1]
        actual = y_train.iloc[test_index]

        fold_results.append(
            {
                "accuracy": accuracy_score(actual, prediction),
                "balanced_accuracy": balanced_accuracy_score(actual, prediction),
                "precision": precision_score(actual, prediction, zero_division=0),
                "recall": recall_score(actual, prediction, zero_division=0),
                "f1_score": f1_score(actual, prediction, zero_division=0),
                "roc_auc": roc_auc_score(actual, probability),
            }
        )
        actual_values.extend(actual)
        predicted_values.extend(prediction)
        probability_values.extend(probability)

    mean_scores = pd.DataFrame(fold_results).mean()
    results.append({"model": model_name, **mean_scores.to_dict()})
    roc_results[model_name] = (actual_values, probability_values)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        confusion_matrix(actual_values, predicted_values),
        annot=True,
        fmt="d",
        cmap="Blues",
    )
    plt.title(model_name)
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.tight_layout()
    plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
for model_name, (actual, probability) in roc_results.items():
    RocCurveDisplay.from_predictions(actual, probability, name=model_name, ax=ax)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_title("ROC Curve Comparison")
fig.tight_layout()
plt.show()

results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
print("5-Fold cross-validation results")
print(results_df.round(4).to_string(index=False))
best_model_name = results_df.iloc[0]["model"]
print("\nBest basic model based on training CV F1-score:", best_model_name)

tree_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("model", models["Decision Tree"])]
)
tree_pipeline.fit(X_train, y_train)

feature_names = tree_pipeline.named_steps["preprocessor"].get_feature_names_out()
importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": tree_pipeline.named_steps["model"].feature_importances_,
    }
)

for column in categorical_columns:
    importance_df.loc[
        importance_df["feature"].str.startswith(f"categorical__{column}_"),
        "feature",
    ] = column
importance_df["feature"] = importance_df["feature"].str.replace(
    "numeric__", "", regex=False
)
importance_df = (
    importance_df.groupby("feature", as_index=False)["importance"]
    .sum()
    .sort_values("importance", ascending=False)
    .head(10)
)

print("\nDecision Tree feature importance")
print(importance_df.round(4).to_string(index=False))

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x="importance", y="feature")
plt.title("Decision Tree Feature Importance")
plt.tight_layout()
plt.show()
