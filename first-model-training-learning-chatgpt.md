I recommend starting with Python + pandas + scikit-learn and building one simple model end-to-end before trying XGBoost or deep learning.

Think of the complete process as:

Dataset → Clean data → Select features → Split data → Train model → Test model → Predict new house prices

Step 1: Create your dataset

Start with a CSV file such as house_prices.csv:

area_sqft	bedrooms	bathrooms	age_years	location	sale_price
1200	2	2	5	Whitefield	9,000,000
1500	3	2	3	Whitefield	12,500,000
1000	2	2	8	Electronic City	6,500,000
1800	3	3	2	HSR Layout	18,000,000
2200	4	3	6	Indiranagar	28,000,000

Here, the model inputs are things such as area, bedrooms, bathrooms, age, and location.

Your target is:

sale_price

The model's job is essentially to learn:

area
bedrooms
bathrooms
age
location
    ↓
 MACHINE LEARNING
    ↓
predicted sale_price
Step 2: Set up Python

For your first project, install:

pip install pandas scikit-learn matplotlib

Then create a Python file or Jupyter notebook.

Import the libraries:

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

Don't worry about understanding every import yet. We'll use each one below.

Step 3: Load the dataset
df = pd.read_csv("house_prices.csv")

print(df.head())

You should see something similar to:

   area_sqft  bedrooms  bathrooms  age_years  location         sale_price
0       1200         2          2          5  Whitefield        9000000
1       1500         3          2          3  Whitefield       12500000
2       1000         2          2          8  Electronic City   6500000

Before training anything, inspect the data:

print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.describe())

You're checking for missing values, incorrect data types, unusual values, and how much data you actually have.

Step 4: Separate inputs and output

Suppose these are the features we want the model to use:

features = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "location"
]

X = df[features]
y = df["sale_price"]

X contains the information given to the model.

y contains the correct answers.

Conceptually:

X                                  y

1200, 2, 2, 5, Whitefield    →    ₹90 lakh
1500, 3, 2, 3, Whitefield    →    ₹1.25 crore
1000, 2, 2, 8, Electronic    →    ₹65 lakh

The model examines many such examples and tries to learn the relationship between X and y.

Step 5: Split the dataset

Never train and evaluate the model using exactly the same houses.

For a simple learning exercise, split your data:

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

That gives you approximately:

1000 houses

        ↓

800 houses                 200 houses
TRAINING                    TESTING

Model learns               Model has never
from these                  seen these

For a real house-price system, I'd eventually switch to a time-based split using the sale date, but this random split is easier for learning the basic workflow.

Step 6: Prepare numeric and categorical columns

Your dataset contains two different kinds of data.

Numeric:

area_sqft
bedrooms
bathrooms
age_years

Categorical:

location

Define them:

numeric_features = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "age_years"
]

categorical_features = [
    "location"
]

Now create preprocessing rules:

numeric_transformer = Pipeline([
    ("missing_values", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("missing_values", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

Then combine them:

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])

This automatically handles missing values and converts text locations into numbers the model can understand.

Step 7: Create the model

For your first experiment, try a Random Forest:

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

Now combine preprocessing and the model:

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

This is convenient because one pipeline handles everything:

Raw house
    ↓
Handle missing values
    ↓
Convert location to numbers
    ↓
Random Forest
    ↓
Price prediction
Step 8: Train the model

This is the actual training step:

pipeline.fit(X_train, y_train)

That's it.

Internally, the algorithm is learning patterns between house characteristics and historical sale prices.

Step 9: Make predictions

Now give the model houses it didn't see during training:

predictions = pipeline.predict(X_test)

For example:

Actual Price       Predicted Price

₹90 lakh           ₹94 lakh
₹1.20 crore        ₹1.14 crore
₹75 lakh           ₹78 lakh
₹2.00 crore        ₹1.82 crore

The predictions won't be perfect. The important question is how wrong they are.

Step 10: Evaluate your model

Start with MAE:

mae = mean_absolute_error(y_test, predictions)

print("MAE:", mae)

Suppose you get:

MAE: 650000

That means the model's predictions are off by roughly ₹6.5 lakh on average on this test set.

You can also calculate RMSE:

rmse = root_mean_squared_error(y_test, predictions)

print("RMSE:", rmse)

RMSE penalizes large prediction errors more strongly.

Step 11: Predict the price of a new house

This is the fun part.

Suppose someone gives you:

Area       = 1600 sqft
Bedrooms   = 3
Bathrooms  = 2
Age        = 4 years
Location   = Whitefield

Create the input:

new_house = pd.DataFrame([{
    "area_sqft": 1600,
    "bedrooms": 3,
    "bathrooms": 2,
    "age_years": 4,
    "location": "Whitefield"
}])

Then predict:

predicted_price = pipeline.predict(new_house)

print(predicted_price[0])

You might get:

13450000

Meaning approximately:

₹1.345 crore

Step 12: Improve the model

Once the basic version works, don't immediately jump to a more complicated algorithm. First improve your data and features.

For house prices, you could add:

latitude
longitude
property_type
floor
total_floors
parking
built_up_area
property_age
distance_to_metro
distance_to_school
distance_to_hospital
amenities
sale_date
locality

For example, your second-generation dataset might be:

                              MODEL INPUTS
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
    PROPERTY                  LOCATION                   MARKET
        │                         │                         │
   area_sqft                  latitude                  sale_date
   bedrooms                   longitude                 market trend
   bathrooms                  locality
   age                        metro distance
   parking
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  ↓
                         House Price Model
                                  ↓
                           Predicted Price

After you understand this pipeline, I would progress in this order:

Linear Regression → Random Forest → XGBoost/LightGBM/CatBoost → feature engineering → hyperparameter tuning → production deployment
