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

# How to Run the Project

## Environment Setup

Python 3.12 is recommended. From the project root directory, create and activate a virtual environment, then install the required libraries.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the Analysis Notebook

The notebook contains the complete analysis workflow and saved plots and model outputs.

```bash
jupyter lab notebook/Hotel_Booking_Cancellation_Project.ipynb
```

Run the notebook cells from top to bottom to reproduce the exploratory analysis, preprocessing summary, classification evaluation, K-means clustering analysis, and final model comparison.

## Run the Source Code Files

The individual Python modules can also be executed from the project root directory in the following order:

```bash
python src/01_DataExploration.py
python src/02_DataPreprocessing.py
python src/03_ClassificationModeling.py
python src/04_KmeansClustering.py
python src/05_ModelComparison.py
```

## Key Outputs

* `src/01_DataExploration.py`: dataset statistics, missing values, cancellation distributions, and EDA plots.
* `src/02_DataPreprocessing.py`: cleaning decisions and preprocessing summary.
* `src/03_ClassificationModeling.py`: baseline classification cross-validation scores, confusion matrices, ROC curves, and feature importance.
* `src/04_KmeansClustering.py`: selected cluster count, cluster summaries, and visualization plots.
* `src/05_ModelComparison.py`: top five preprocessing/model combinations, selected model, and held-out test set scores.
* `notebook/Hotel_Booking_Cancellation_Project.ipynb`: consolidated executable report containing outputs and plots.

---

# Project Structure

```text
DataScience_HotelBooking/
|
|-- data/
|       `-- hotel_bookings.csv
|
|-- notebook/
|       `-- Hotel_Booking_Cancellation_Project.ipynb
|
|-- src/
|   |-- ProjectUtils.py
|   |-- 01_DataExploration.py
|   |-- 02_DataPreprocessing.py
|   |-- 03_ClassificationModeling.py
|   |-- 04_KmeansClustering.py
|   `-- 05_ModelComparison.py
|
|-- library.md
|
|-- requirements.txt
|
`-- README.md
```
