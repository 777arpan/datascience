# House Price Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Load Dataset
df = pd.read_csv('AmesHousing.csv')

# Select relevant features
selected_columns = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
                    'FullBath', 'YearBuilt', 'YearRemodAdd', 'LotArea', 'Neighborhood', 'SalePrice']
df = df[selected_columns]

# Handle missing values
df = df.dropna()

# Feature Engineering
df['HouseAge'] = 2025 - df['YearBuilt']
df['RemodAge'] = 2025 - df['YearRemodAdd']

# One-hot encode categorical variable (Neighborhood)
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
neighborhood_encoded = ohe.fit_transform(df[['Neighborhood']])
neighborhood_df = pd.DataFrame(neighborhood_encoded, columns=ohe.get_feature_names_out(['Neighborhood']))

# Final dataset
X = pd.concat([df.drop(['SalePrice', 'Neighborhood', 'YearBuilt', 'YearRemodAdd'], axis=1).reset_index(drop=True), neighborhood_df], axis=1)
y = df['SalePrice']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R^2 Score: {r2_score(y_test, y_pred):.2f}")

# Feature Importance
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.grid(True)
plt.tight_layout()
plt.show()
