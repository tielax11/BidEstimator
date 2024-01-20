import json
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
with open('training_data/basic.json', 'r') as f:
    data = json.load(f)

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Convert the 'Tag' lists into dummy variables
mlb = MultiLabelBinarizer()
tag_dummies = pd.DataFrame(mlb.fit_transform(df.pop('Tag')), columns=mlb.classes_, index=df.index)
df = df.join(tag_dummies.add_prefix('Tag_'))

# Convert the remaining categorical variables to dummy variables
df = pd.get_dummies(df, columns=['Artist Level', 'Type', 'Level of Detail'])

# Split the data into input X and output y
X = df.drop('Bid Days', axis=1)
y = df['Bid Days']

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Print the coefficients for each category in the desired format
coefficients = {
    'Level of Detail': dict(zip(df.columns[df.columns.str.startswith('Level of Detail_')], model.coef_)),
    'Type': dict(zip(df.columns[df.columns.str.startswith('Type_')], model.coef_)),
    'Artist Level': dict(zip(df.columns[df.columns.str.startswith('Artist Level_')], model.coef_)),
    'Tag': dict(zip(df.columns[df.columns.str.startswith('Tag_')], model.coef_))
}
print(json.dumps(coefficients, indent=4))

# Load the true values
with open('examples/basic_weights.json', 'r') as f:
    true_values = json.load(f)

# Compute the accuracy as the mean squared error between the learned and true values
accuracy = mean_squared_error([true_values[key][subkey] for key in true_values for subkey in true_values[key]],
                              [coefficients[key][subkey] for key in coefficients for subkey in coefficients[key]])
print('Accuracy:', accuracy)