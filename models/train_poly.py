import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MultiLabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import json

# Load the data
df = pd.read_json('training_data/basic.json')

# Separate the DataFrame into features (X) and target (y)
X = df.drop('Bid Days', axis=1)
y = df['Bid Days']

# One-hot encode the list feature 'Tag'
mlb = MultiLabelBinarizer()
expandedTagData = mlb.fit_transform(X['Tag'])
tagClasses = mlb.classes_

expandedTags = pd.DataFrame(expandedTagData, columns=tagClasses)
X = pd.concat([X.drop('Tag', axis=1), expandedTags], axis=1)

# One-hot encode the remaining categorical features
X = pd.get_dummies(X)

# Create a PolynomialFeatures object
poly = PolynomialFeatures(degree=2)

# Transform the input features
X_poly = poly.fit_transform(X)

# Create a LinearRegression object
model = LinearRegression()

# Fit the model to the data
model.fit(X_poly, y)

# Get the coefficients
coefficients = model.coef_

# Predict the target values using the model
y_pred = model.predict(X_poly)

# Print the predicted data
print('Predicted data:', y_pred)

# Compute the accuracy as the mean squared error between the true and predicted values
accuracy = mean_squared_error(y, y_pred)
print('Accuracy:', accuracy)