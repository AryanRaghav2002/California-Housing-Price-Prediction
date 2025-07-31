import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

# Copying the model and pipeling if they exist.
MODEL_FILE="model.pkl"
PIPLINE_FILE="pipline.pkl"


def build_pipeline(num_attributes, cat_attributes):

    # Constructing numerical pipleine
    num_pipeline=Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    # contructing categorical pipeline
    cat_pipeline=Pipeline([
        ("Encoding", OneHotEncoder(handle_unknown="ignore"))
    ])

    # constructing full pipeline
    full_pipeline=ColumnTransformer([
        ("num", num_pipeline, num_attributes),
        ("cat", cat_pipeline, cat_attributes)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Then we train the model
    
    # Load the dataset
    housing=pd.read_csv("housing.csv")

    # Split the dataset into training and testing data
    housing["income_cat"]=pd.cut(housing["median_income"], bins=[0,1.5,3.0,4.5,6.0,np.inf], labels=[1,2,3,4,5])

    Split=StratifiedShuffleSplit(test_size=0.2, random_state=42, n_splits=1)
    for train_index, test_index in Split.split(housing, housing["income_cat"]):
        housing.loc[test_index].drop("income_cat", axis=1).to_csv("input.csv", index=False)
        housing=housing.loc[train_index].drop("income_cat", axis=1)
        
    # Seperate features and labels
    housing_features=housing.drop("median_house_value", axis=1)
    housing_labels=housing["median_house_value"].copy()

    # List numerical and categorical values
    num_attribs=housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs=["ocean_proximity"]

    # Build pipeline and passing the data thru it
    pipeline=build_pipeline(num_attribs, cat_attribs)
    housing_prepared=pipeline.fit_transform(housing_features)

    model=RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    # Save the model
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPLINE_FILE)

    print("Model trained successfully and, new model and pipeline file is created.")

else:

    # Loading the already existing model and pipleine file
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPLINE_FILE)

    # Making predictions on test_data/new_data
    input_data=pd.read_csv("input.csv")
    transformed_input=pipeline.transform(input_data)
    predictions=model.predict(transformed_input)
    input_data["new_median_house_value"]=predictions

    # Making a output data file to compare the results.
    input_data.to_csv("output_data.csv", index=False)

    print("Inference is complete result is saved to output_data.")
