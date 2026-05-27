# Hotel Booking Cancellation Prediction

Predicting whether a hotel reservation will be canceled, using the [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) dataset (about 119k records, 32 features). The target column is `is_canceled`.

## What's in here

The whole analysis lives in one notebook -> `notebook/Hotel_Booking_Cancellation.ipynb`.
It walks through EDA, cleaning, classification with K-fold CV, K-means, and a top-level comparison function that sweeps preprocessing and model combinations. The same code is also split into numbered scripts under `src/` for graders who prefer to run things one step at a time.

There's also a small bonus web demo (`06_practical_demo.py`) that wraps the final model in a local web page. See the bottom of this file.

## Data cleaning

A few decisions worth flagging up front, because they affect the row count:

- Missing `children` is filled with 0 (treated as no children listed).
- Missing `country`, `agent`, `company` get explicit "Unknown" labels rather than being dropped.
- Rows with zero total guests, zero total stay nights, or negative ADR are removed. These looked like data-entry errors or test bookings rather than real reservations.
- Duplicate rows are **kept**. The dataset has no booking ID, so two identical rows could just as easily be two real bookings with the same attributes as one accidental duplicate. We didn't feel comfortable deleting tens of thousands of rows on a guess.
- Columns that leak the outcome are dropped before modeling: `reservation_status`, `reservation_status_date`, and `assigned_room_type` (assigned_room_type may reflect post-booking operational decisions and therefore can leak information unavailable at the original reservation time).
- Three engineered columns: `total_guests`, `total_stays`, `is_family`.

After cleaning the cancellation rate is basically unchanged from the raw data, which is the behavior we wanted.

## Models and evaluation

Classification models compared: Logistic Regression, Decision Tree, K-Nearest Neighbors. Each is tested with multiple parameter settings (e.g. `C=0.1` vs `1.0`, `max_depth=3/5/8`, `n_neighbors=5/11/21`).

Preprocessing variants tried per model: `{StandardScaler, MinMaxScaler}` × `{OneHotEncoder, OrdinalEncoder}`. That gives 4 preprocessing combinations × 8 model configurations = 32 combinations total, each scored with stratified 5-fold cross-validation.

For clustering we use K-means on the cleaned numerical features. Elbow and silhouette are both plotted in Section 4 of the notebook.

Reported metrics: accuracy, balanced accuracy, precision, recall, F1, and ROC-AUC for the final model on the test set.

### About the 20,000-row sample

The comparison function in Section 5 uses a stratified 20,000-row sample rather than the full cleaned dataset. This keeps the 32-combination sweep tractable (KNN with one-hot-encoded high-cardinality columns gets slow fast). Section 5.1 retrains the selected best model on the full training set and re-evaluates it as a sanity check — the scores stay in the same ballpark, so the sample is fine for model selection.

## Results

Best combination from the sweep: **StandardScaler + OneHotEncoder + Logistic Regression (C=1.0, class_weight=balanced)**.

On the test set the model lands around 0.83 accuracy and 0.79 F1, with ROC-AUC about 0.92. Exact numbers per metric (accuracy, balanced accuracy, precision, recall, F1, ROC-AUC) are printed at the end of Section 5 in the notebook.

As a stability check we retrained the same configuration on the full training set and evaluated it on a separate test set. The scores stayed close to the sample-based ones, confirming that the sample-based selection generalizes.

## Project structure

```
DataScience_HotelBooking/
├── data/
│   ├── raw/
│   └── hotel_bookings.csv
├── notebook/
│   └── Hotel_Booking_Cancellation.ipynb
├── src/
│   ├── project_utils.py
│   ├── 01_data_exploration.py
│   ├── 02_data_preprocessing.py
│   ├── 03_classification_modeling.py
│   ├── 04_kmeans_clustering.py
│   ├── 05_open_source_model_comparison.py
│   └── 06_practical_demo.py
├── library.md
└── README.md
```

`library.md` lists every non-trivial library, class, and method used in the project with a short explanation, per the term project rubric.

## Environment

Tested with:
- Python 3.14.2
- pandas 3.0.1
- scikit-learn 1.8.0

## References

- Dataset: Antonio, Almeida, & Nunes (2019). *Hotel booking demand datasets.* Data in Brief, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126
- Kaggle mirror: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

Code patterns for the preprocessing comparison function are loosely based on the scikit-learn `Pipeline` + `ColumnTransformer` examples in the official user guide.

## Bonus demo

`06_practical_demo.py` is a lightweight local demo that loads the cleaned dataset, trains the final model, and provides cancellation predictions through a localhost web interface

This optional demo was created to make the prediction results easier to interpret during the presentation.
