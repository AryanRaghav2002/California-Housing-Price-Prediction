# 🏠 California Housing Price Prediction

This project uses machine learning to predict housing prices in California based on various features such as median income, location, etc.
It is built using Python and Scikit-learn and includes end-to-end steps from data cleaning to model deployment preparation.

---

## 📁 Project Structure
California-Housing-Price-Prediction/
│
├── housing.csv # Raw dataset
├── housing.pkl # Trained ML model (Pickle)
├── housing_model.joblib # Trained ML model (Joblib)
├── predictor.py # Python script for model prediction
├── requirements.txt # Project dependencies
├── README.md # Project overview
└── ... # Jupyter notebooks, EDA files, etc.

---

## 📊 Dataset

The dataset is derived from the **California Housing Dataset** available via `sklearn.datasets.fetch_california_housing`.  
Alternatively, it was saved as `housing.csv` for local access.

**Features include:**
- `longitude`, `latitude`
- `housing_median_age`
- `total_rooms`, `total_bedrooms`
- `population`, `households`
- `median_income`
- `median_house_value` (target)

---

## ⚙️ Technologies Used

- **Python 3.10+**
- **scikit-learn**
- **pandas**, **numpy**
- **matplotlib**, **seaborn**
- **pickle**, **joblib**

---

## 📈 Model Info
The project uses a Linear Regression model initially.
Later versions may include Decision Trees or Random Forests.
Model performance is evaluated using Root Mean Squared Error (RMSE) and Cross-Validation.

## 🧠 Key Concepts
Data Cleaning
Exploratory Data Analysis (EDA)
Feature Scaling
Train-Test Split
Model Training & Evaluation
Saving/Loading model using pickle / joblib

## 📌 Future Enhancements
Add a web interface using Flask or Streamlit
Use grid search or randomized search for hyperparameter tuning
