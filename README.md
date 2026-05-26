# Hotel Booking Cancellation Prediction

## Overview

This project analyzes hotel booking data to predict whether a reservation will be canceled using machine learning models.

We also applied K-means clustering to explore booking patterns in the dataset.

The project includes:

* Exploratory Data Analysis (EDA)
* Data preprocessing
* Feature engineering
* Classification modeling
* Clustering analysis
* Model evaluation

---

# Dataset

This project uses the **Hotel Booking Demand** dataset, which contains hotel reservation records from Portugal.

## Dataset Information

* Number of records: 119,390
* Number of features: 32
* Target variable: `is_canceled`

The dataset contains:

* Numerical variables
* Categorical variables
* Missing values
* Duplicate rows
* Some suspicious records

Dataset sources:

* https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
* https://doi.org/10.1016/j.dib.2018.11.126

---

# Models

Classification models:

* Logistic Regression
* Decision Tree
* K-Nearest Neighbors (KNN)

Clustering model:

* K-means clustering

Evaluation methods:

* Stratified 5-Fold Cross Validation
* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

---

# Data Preprocessing

Main preprocessing steps:

* Missing value handling
* Feature engineering
* Feature scaling
* Categorical encoding
* Leakage prevention

Created features:

* `total_guests`
* `total_stays`
* `is_family`

Removed columns:

* `reservation_status`
* `reservation_status_date`
* `assigned_room_type`

Scaling methods:

* StandardScaler
* MinMaxScaler

Encoding methods:

* OneHotEncoder
* OrdinalEncoder

---

# Model Comparison

A reusable function was created to compare different preprocessing methods and machine learning models.

The comparison includes:

* Different scaling methods
* Different encoding methods
* Multiple classification models
* Different parameter settings

The results are compared using F1-score and balanced accuracy.

---

# Project Structure

```text
DataScience_HotelBooking/
|
|-- data/
|       `-- hotel_bookings.csv
|
|-- src/
|   |-- project_utils.py
|   |-- 01_data_exploration.py
|   |-- 02_data_preprocessing.py
|   |-- 03_classification_modeling.py
|   |-- 04_kmeans_clustering.py
|   `-- 05_open_source_model_comparison.py
|
`-- README.md
```
