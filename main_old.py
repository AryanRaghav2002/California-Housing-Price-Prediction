import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1. Load the dataset
housing=pd.read_csv("housing.csv")

# 2. Creating a Stratified Test set
housing["income_cat"]=pd.cut(housing["median_income"], bins=[0,1.5,3.0,4.5,6.0,np.inf], labels=[1,2,3,4,5])
Split=StratifiedShuffleSplit(test_size=0.2, random_state=42, n_splits=1)

for train_index, test_index in Split.split(housing, housing["income_cat"]):
    Strat_train_set=housing.loc[train_index].drop("income_cat", axis=1)
    Strat_test_set=housing.loc[test_index].drop("income_cat", axis=1)

# We will work on traing data
housing=Strat_train_set.copy()

# 3. Seperate features and labels
housing_labels=housing["median_house_value"].copy() #It is our label
housing=housing.drop("median_house_value", axis=1) # We will work on this data

##print(housing, housing_labels)

# 4. List the numerical and categorical values
num_attribs=housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs=["ocean_proximity"]

# 5. Make the pipeline 
# for numerical columns
num_pipline=Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# for categorical columns
cat_pipline=Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore")) # i.e if new value comes it will be ignored
])

# Construct full pipline
full_pipeline=ColumnTransformer([
    ("num", num_pipline, num_attribs),
    ("cat", cat_pipline, cat_attribs)
])

# 6. Transform the data using full pipline(column tranformmer)
housing_prepared=full_pipeline.fit_transform(housing)
print(housing_prepared.shape)


# 7. Training the model

# Linear Regression model
lin_reg_model=LinearRegression()
lin_reg_model.fit(housing_prepared, housing_labels)
lin_reg_pred=lin_reg_model.predict(housing_prepared)
# lin_reg_rmse=root_mean_squared_error(housing_labels, lin_reg_pred)
lin_reg_rmses= -cross_val_score(lin_reg_model, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10
                              )
# print(f"The root mean squared error for Linear Regression is {lin_reg_rmse}")
print(f"After Cross validation Linear rmse: {pd.Series(lin_reg_rmses).mean()}") # Since it runs on multiple data((k-1) i.e 9)


# Decision Tree Regression model
dec_reg_model=DecisionTreeRegressor()
dec_reg_model.fit(housing_prepared, housing_labels)
dec_reg_pred=dec_reg_model.predict(housing_prepared)
# dec_reg_rmse=root_mean_squared_error(housing_labels, dec_reg_pred)
dec_reg_rmses= -cross_val_score(dec_reg_model, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10
                              )
# print(f"The root mean squared error for Decision Tree Regressor is {dec_reg_rmse}")
print(f"After Cross validation Decision tree rmse: {pd.Series(dec_reg_rmses).mean()}") # Since it runs on multiple data((k-1) i.e 9)


# Random forest regressor model
random_forest_reg_model=RandomForestRegressor()
random_forest_reg_model.fit(housing_prepared, housing_labels)
random_forest_reg_pred=random_forest_reg_model.predict(housing_prepared)
# random_forest_reg_rmse=root_mean_squared_error(housing_labels, random_forest_reg_pred)
random_forest_reg_rmses= -cross_val_score(random_forest_reg_model, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10
                              )
# print(f"The root mean squared error for Random Forest regressor is {random_forest_reg_rmse}")

# Output:
#         (16512, 13)
#     After Cross validation Linear rmse: 69204.32275494764
#     After Cross validation Decision tree rmse: 69318.85112765571
#     After Cross validation Random forest rmse: 49393.923744988846

print(f"After Cross validation Random forest rmse: {pd.Series(random_forest_reg_rmses).mean()}") # Since it runs on multiple data((k-1) i.e 9)
