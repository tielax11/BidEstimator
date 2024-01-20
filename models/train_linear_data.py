import json
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from scipy.optimize import nnls

# Load the data
with open('training_data/basic.json', 'r') as f:
    data = json.load(f)

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Convert the 'Tag' lists into dummy variables
mlb = MultiLabelBinarizer()
tag_dummies = pd.DataFrame(mlb.fit_transform(df.pop('Tag')), columns=mlb.classes_, index=df.index)
df = df.join(tag_dummies.add_prefix('Tag_'))

# Convert 'Artist Level', 'Type', and 'Level of Detail' to dummy variables
df = pd.get_dummies(df, columns=['Artist Level', 'Type', 'Level of Detail'])

# Calculate the sum of all tags
df['Sum_Tags'] = df[df.columns[df.columns.str.startswith('Tag_')].tolist()].sum(axis=1)

# Calculate Bid Days according to the formula
df['Calculated_Bid_Days'] = df['Sum_Tags'] * df[df.columns[df.columns.str.startswith('Artist Level_')].tolist()].sum(axis=1) * df[df.columns[df.columns.str.startswith('Type_')].tolist()].sum(axis=1) * df[df.columns[df.columns.str.startswith('Level of Detail_')].tolist()].sum(axis=1)

# Now you can use 'Calculated_Bid_Days' as a feature in your model
X = df.drop(['Bid Days', 'Calculated_Bid_Days'], axis=1)
y = df['Calculated_Bid_Days']

# Assume X is your feature matrix
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

class NonNegativeLinearRegression(LinearRegression):
    def fit(self, X, y):
        self.coef_, self.intercept_ = nnls(X, y)
        return self

# Train a non-negative linear regression model
model = NonNegativeLinearRegression()
model.fit(X_scaled, y)

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