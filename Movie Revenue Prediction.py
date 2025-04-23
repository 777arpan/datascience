# Movie Revenue Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import json

# Load Dataset
df = pd.read_csv('tmdb_5000_movies.csv')

# Drop rows with missing or zero budget/revenue
df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
df.dropna(subset=['release_date'], inplace=True)

# Convert release_date to datetime
df['release_date'] = pd.to_datetime(df['release_date'])
df['release_month'] = df['release_date'].dt.month

# Extract genres from JSON string
def extract_genres(genre_str):
    try:
        genres = json.loads(genre_str.replace("'", '"'))
        return [g['name'] for g in genres]
    except:
        return []

df['genres_list'] = df['genres'].apply(extract_genres)

# One-hot encode genres
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
genre_encoded = ohe.fit_transform(df['genres_list'].apply(lambda x: ','.join(x)).values.reshape(-1, 1))
genre_df = pd.DataFrame(genre_encoded, columns=ohe.get_feature_names_out(['genre']))
df = pd.concat([df.reset_index(drop=True), genre_df], axis=1)

# Select features and target
features = ['budget', 'popularity', 'runtime', 'release_month'] + list(genre_df.columns)
X = df[features]
y = df['revenue']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R^2 Score: {r2_score(y_test, y_pred):.2f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Revenue')
plt.ylabel('Predicted Revenue')
plt.title('Actual vs Predicted Revenue')
plt.grid(True)
plt.tight_layout()
plt.show()
