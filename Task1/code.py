import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("HousingData.csv")  
print("Initial Data Sample:\n", df.head())

# Replace 'NA' strings with np.nan for numerical handling
df.replace("NA", np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df = df[df["MEDV"].notna()]

# Fill missing values in features with mean
imputer = SimpleImputer(strategy="mean")
X = df.drop("MEDV", axis=1)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
y = df["MEDV"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Mean Squared Error: {mse:.2f}")
print(f"✅ R² Accuracy Score: {r2*100:.2f}%")
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Boston House Prices")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.grid(True)
plt.show()
