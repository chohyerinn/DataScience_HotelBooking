import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from project_utils import load_raw_data

df = load_raw_data()

numerical_columns = [
    "lead_time",
    "adr",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "booking_changes",
    "total_of_special_requests",
]
rate_columns = ["hotel", "deposit_type", "customer_type", "market_segment"]

print("Dataset shape:", df.shape)
print("\nNumerical summary")
print(df[numerical_columns].describe().T.round(2))

print("\nMissing values")
print(df.isnull().sum().loc[lambda values: values > 0].sort_values(ascending=False))
print("\nDuplicate rows:", df.duplicated().sum())

total_guests = df["adults"] + df["children"].fillna(0) + df["babies"]
total_stays = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
print("Zero total guests:", (total_guests == 0).sum())
print("Zero total stay nights:", (total_stays == 0).sum())
print("Negative ADR:", (df["adr"] < 0).sum())

q1 = df["adr"].quantile(0.25)
q3 = df["adr"].quantile(0.75)
iqr = q3 - q1
adr_outliers = df[(df["adr"] < q1 - 1.5 * iqr) | (df["adr"] > q3 + 1.5 * iqr)]
print("ADR outliers by IQR:", len(adr_outliers))

target_summary = pd.DataFrame(
    {
        "count": df["is_canceled"].value_counts().sort_index(),
        "percentage": (
            df["is_canceled"].value_counts().sort_index() / len(df) * 100
        ).round(2),
    }
)
print("\nCancellation distribution")
print(target_summary)
print(
    "\nCancellation is about 37%, so it is not severely imbalanced, "
    "but stratified validation is appropriate."
)

for column in rate_columns:
    rates = df.groupby(column)["is_canceled"].mean().mul(100).round(2)
    print(f"\nCancellation rate by {column}")
    print(rates)

lead_time_groups = pd.cut(
    df["lead_time"],
    bins=[-1, 7, 30, 90, 180, 365, float("inf")],
    labels=["0-7", "8-30", "31-90", "91-180", "181-365", "366+"],
)
lead_time_rate = (
    df.assign(lead_time_group=lead_time_groups)
    .groupby("lead_time_group", observed=False)["is_canceled"]
    .mean()
    .mul(100)
    .round(2)
)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="is_canceled")
plt.title("Booking Cancellation Distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
correlation = df[numerical_columns + ["is_canceled"]].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Numerical Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x=lead_time_rate.index, y=lead_time_rate.values)
plt.title("Cancellation Rate by Lead Time Group")
plt.xlabel("Lead Time (Days)")
plt.ylabel("Cancellation Rate (%)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, y="adr")
plt.title("ADR Distribution")
plt.ylabel("Average Daily Rate (ADR)")
plt.tight_layout()
plt.show()
