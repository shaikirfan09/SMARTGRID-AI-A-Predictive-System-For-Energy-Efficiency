# =========================================
# SMARTGRID-AI: Predictive System for Energy Efficiency
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

print("\nLoading dataset...")

# Load dataset
df = pd.read_csv("energy consumption data.csv")

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

# ================================
# CHECK MISSING VALUES
# ================================

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# ================================
# CONVERT TIME COLUMN
# ================================

df['Datetime'] = pd.to_datetime(df['time'], errors='coerce', utc=True)

# ================================
# FEATURE ENGINEERING
# ================================

df['hour'] = df['Datetime'].dt.hour
df['day_of_week'] = df['Datetime'].dt.dayofweek
df['month'] = df['Datetime'].dt.month

df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
df['is_night'] = df['hour'].apply(lambda x: 1 if x >= 20 or x <= 6 else 0)

print("\n===== NEW FEATURES CREATED =====")
print(df[['hour','day_of_week','month','is_weekend','is_night']].head())

# ================================
# TARGET COLUMN
# ================================

target_col = "total load actual"

df[target_col] = df[target_col].fillna(df[target_col].median())

# ================================
# DEFINE FEATURES
# ================================

features = ['hour','day_of_week','month','is_weekend','is_night']

X = df[features]
y = df[target_col]

# ================================
# TRAIN TEST SPLIT
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# TRAIN MODEL
# ================================

print("\nTraining Random Forest Model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ================================
# PREDICTIONS
# ================================

predictions = model.predict(X_test)

# ================================
# MODEL PERFORMANCE
# ================================

mae = mean_absolute_error(y_test, predictions)

print("\n===== MODEL PERFORMANCE =====")
print(f"Average Prediction Error (MAE): {mae:.2f} MW")

# ================================
# PERFORMANCE GRAPH
# ================================

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
}).reset_index(drop=True)

plt.figure(figsize=(12,6))

plt.plot(results["Actual"][:100], label="Actual Demand")
plt.plot(results["Predicted"][:100], linestyle="--", label="Predicted Demand")

plt.title("Energy Demand: Actual vs Predicted")
plt.xlabel("Hours")
plt.ylabel("Energy Units (MW)")
plt.legend()
plt.grid(True)

plt.savefig("model_performance.png")

plt.show()

# ================================
# ANOMALY DETECTION
# ================================

results["Error"] = results["Actual"] - results["Predicted"]

threshold = results["Predicted"] * 0.15

anomalies = results[results["Error"] > threshold]

print("\n===== ENERGY WASTE DETECTION =====")
print("Number of anomaly hours detected:", len(anomalies))

# ================================
# FEATURE IMPORTANCE
# ================================

importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,4))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance_df
)

plt.title("Feature Importance for Energy Consumption")

plt.savefig("feature_importance.png")

plt.show()

# ================================
# SUSTAINABILITY REPORT
# ================================

total_wasted_mw = anomalies["Error"].sum()

co2_saved_kg = total_wasted_mw * 1000 * 0.19
co2_saved_tonnes = co2_saved_kg / 1000

print("\n===== SUSTAINABILITY REPORT =====")

print(f"Total potential energy waste detected: {total_wasted_mw:.2f} MWh")
print(f"Estimated CO2 reduction: {co2_saved_tonnes:.2f} tonnes")
print(f"Equivalent trees planted: {int(co2_saved_tonnes * 45)}")

print("\nGraphs saved:")
print("model_performance.png")
print("feature_importance.png")