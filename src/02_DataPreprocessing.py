import pandas as pd

from project_utils import LEAKAGE_COLUMNS, clean_booking_data, load_raw_data


raw_df = load_raw_data()
cleaned_df = clean_booking_data(raw_df)

summary = pd.Series(
    {
        "raw_rows": len(raw_df),
        "cleaned_rows": len(cleaned_df),
        "removed_invalid_rows": len(raw_df) - len(cleaned_df),
        "raw_cancellation_rate_percent": raw_df["is_canceled"].mean() * 100,
        "cleaned_cancellation_rate_percent": cleaned_df["is_canceled"].mean() * 100,
        "duplicate_rows_kept_for_review": raw_df.duplicated().sum(),
        "remaining_zero_night_bookings": (cleaned_df["total_stays"] == 0).sum(),
        "remaining_missing_values": cleaned_df.isnull().sum().sum(),
    }
)

print("Preprocessing summary")
print(summary.round(2))

print("\nCleaning decisions")
print("- Filled missing children values with 0")
print("  Missing children values are treated as no children in the reservation.")
print("- Filled missing country, agent, and company values with labels")
print("- Created total_guests, total_stays, and is_family")
print("- Removed rows with zero guests or negative ADR")
print("- Kept identical rows because there is no booking ID to prove duplication")
print("- Dropped leakage columns:", LEAKAGE_COLUMNS)
print("- Encoding and scaling are applied in the modeling pipeline")
