# California Housing Price Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning project that predicts median house values across California census block groups using statistical modeling, geospatial data analysis, and ensemble regression techniques.

---

## 📌 Project Overview

Accurate housing price estimation is vital for buyers, sellers, real estate developers, and financial institutions. This project leverages the **California Housing Dataset** to build and compare multiple regression models capable of predicting median house prices based on demographic, geographical, and structural features.

### Key Highlights
- **Exploratory Data Analysis (EDA):** Comprehensive distribution and correlation analysis to uncover key drivers of property values (e.g., median income, location).
- **Geospatial Visualization:** Mapped spatial relationships between geographic coordinates (Latitude/Longitude) and housing values.
- **Data Preprocessing & Scaling:** Feature scaling using `StandardScaler` to ensure optimal performance across regression models.
- **Model Benchmarking:** Evaluated and compared **Linear Regression**, **Decision Tree Regressor**, and **Random Forest Regressor** using robust regression metrics ($R^2$, MAE, MSE).
- **Inference Pipeline:** Included custom prediction script to estimate house values for unseen sample data.

---

## 📊 Dataset Description

The dataset is sourced from the **StatLib repository** and distributed via Scikit-Learn (`fetch_california_housing`), derived from the 1990 U.S. Census.

| Feature | Description |
| :--- | :--- |
| `MedInc` | Median income in block group (in tens of thousands of USD) |
| `HouseAge` | Median house age in block group (in years) |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Total block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group centroid latitude |
| `Longitude` | Block group centroid longitude |
| **`MedHouseVal` (Target)** | Median house value (in hundreds of thousands of USD, $100k) |

---

## 🛠️ Architecture & Workflow

```mermaid
graph TD
    A[Data Ingestion: California Housing Dataset] --> B[Exploratory Data Analysis & Geospatial Plotting]
    B --> C[Feature Preprocessing & Standard Scaling]
    C --> D[Train-Test Split (70/30)]
    D --> E1[Linear Regression]
    D --> E2[Decision Tree Regressor]
    D --> E3[Random Forest Regressor]
    E1 --> F[Model Evaluation: MAE, MSE, R² Score]
    E2 --> F
    E3 --> F
    F --> G[Best Model Selection & Inference on New Data]
```

---

## 📈 Model Performance & Results

The models were evaluated on an unseen test set (30% split) using the following metrics:

| Model | MAE | MSE | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | ~0.53 | ~0.53 | ~0.60 |
| **Decision Tree Regressor** | ~0.46 | ~0.50 | ~0.62 |
| **Random Forest Regressor** | **~0.33** | **~0.25** | **~0.81** |

> **Key Finding:** The **Random Forest Regressor** outperformed other models, explaining over **80% of the variance** ($R^2 \approx 0.81$) by effectively capturing non-linear relationships and interactions between geospatial attributes and median income.

---

## 📂 Project Structure

```text
├── House value prediction.ipynb    # Jupyter Notebook with step-by-step EDA & modeling
├── House value prediction.py       # Python script for end-to-end execution & inference
├── requirements.txt                # Required Python packages
├── README.md                       # Project documentation
└── .gitignore                      # Ignored files (virtual environments, checkpoints)
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/irohitgangwar/house-price-prediction.git
cd house-price-prediction
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline
To execute the complete pipeline and view predictions:
```bash
python "House value prediction.py"
```
Or launch the interactive Jupyter Notebook:
```bash
jupyter notebook "House value prediction.ipynb"
```

---

## 🔮 Predicting on New Data

You can pass new sample data through the trained pipeline:

```python
import numpy as np
import pandas as pd

# Sample house profile
sample_data = pd.DataFrame([{
    'MedInc': 7.325,
    'HouseAge': 30.0,
    'AveRooms': 5.984,
    'AveBedrms': 1.0238,
    'Population': 280.0,
    'AveOccup': 2.20,
    'Latitude': 37.88,
    'Longitude': -122.23
}])

# Scale features and predict
scaled_sample = scaler.transform(sample_data)
predicted_value = rforest.predict(scaled_sample)

print(f"Predicted Median House Value: ${predicted_value[0] * 100000:,.2f}")
```

---

## 🛠️ Tech Stack & Libraries
- **Language:** Python 3.x
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`

---

## 👤 Author

**Rohit Gangwar**
- GitHub: [@irohitgangwar](https://github.com/irohitgangwar)
- Email: irohitgangwar@gmail.com
