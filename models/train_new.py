import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Load the data
with open('training_data/basic.json', 'r') as f:
    data = json.load(f)

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Explode the 'Tag' column
df = df.explode('Tag')

# Perform one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Level of Detail', 'Type', 'Artist Level', 'Tag'])

# Group by index (to handle exploded 'Tag' column) and calculate the sum
df_encoded = df_encoded.groupby(df_encoded.index).sum()

# Take the logarithm of 'Bid Days' to transform the equation into a linear one
df_encoded['Log Bid Days'] = np.log(df_encoded['Bid Days'])

# Prepare the data for the model
X = df_encoded.drop(['Bid Days', 'Log Bid Days'], axis=1).values
y = df_encoded['Log Bid Days'].values

# Fit a Ridge Regression model
model = Ridge(alpha=1.0).fit(X, y)

# Print the weights
predicted_weights = {feature: np.exp(weight) for feature, weight in zip(df_encoded.drop(['Bid Days', 'Log Bid Days'], axis=1).columns, model.coef_)}

# Load the actual weights
with open('examples/basic_weights.json', 'r') as f:
    actual_weights = json.load(f)

# Flatten the actual weights
actual_weights_flat = {f'{category}_{key}': value for category, weights in actual_weights.items() for key, value in weights.items()}

# Calculate the accuracy for each feature
accuracy = {feature: min(weight / actual_weights_flat[feature], actual_weights_flat[feature] / weight) for feature, weight in predicted_weights.items()}

# Print the weights and accuracy for each feature
accuracies = []
for feature, weight in predicted_weights.items():
    weight = round(weight, 1)
    accuracy_percentage = round(accuracy[feature] * 100, 1)
    accuracies.append(accuracy_percentage)
    print(f'{feature}: {weight}, Accuracy: {accuracy_percentage}%')

# Calculate and print the overall accuracy
overall_accuracy = round(sum(accuracies) / len(accuracies), 1)
print(f'Overall Accuracy: {overall_accuracy}%')