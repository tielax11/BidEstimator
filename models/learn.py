import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the data
with open('training_data/basic.json', 'r') as f:
    data = json.load(f)

# Convert the data to a pandas DataFrame
df = pd.json_normalize(data)

# Prepare the features and target variable
X = df.drop('Bid Days', axis=1)
y = df['Bid Days']

# One-hot encode the categorical features
ct = ColumnTransformer(
    [("one_hot_encoder", OneHotEncoder(), list(X.columns))], 
    remainder="passthrough"
)
X = ct.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Print the weights (coefficients) of the model
print(model.coef_)