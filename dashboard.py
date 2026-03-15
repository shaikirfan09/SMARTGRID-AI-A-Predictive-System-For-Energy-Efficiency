import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="SMARTGRID-AI", layout="wide")

st.title("⚡ SMARTGRID-AI Energy Efficiency Dashboard")

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("energy consumption data.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ==========================
# FIX TIME COLUMN
# ==========================

df['Datetime'] = pd.to_datetime(df['time'], errors='coerce', utc=True)

# ==========================
# FEATURE ENGINEERING
# ==========================

df['hour'] = df['Datetime'].dt.hour
df['day_of_week'] = df['Datetime'].dt.dayofweek
df['month'] = df['Datetime'].dt.month

df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
df['is_night'] = df['hour'].apply(lambda x: 1 if x >= 20 or x <= 6 else 0)

# ==========================
# TARGET COLUMN
# ==========================

target_col = "total load actual"

df[target_col] = df[target_col].fillna(df[target_col].median())

features = ['hour','day_of_week','month','is_weekend','is_night']

X = df[features]
y = df[target_col]

# ==========================
# TRAIN MODEL
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

st.subheader("Model Performance")

st.metric("Average Prediction Error (MAE)", round(mae,2))

# ==========================
# ACTUAL VS PREDICTED GRAPH
# ==========================

st.subheader("Energy Demand: Actual vs Predicted")

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
}).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(results["Actual"][:100], label="Actual Demand")
ax.plot(results["Predicted"][:100], linestyle="--", label="Predicted Demand")

ax.set_title("Energy Demand Comparison")
ax.set_xlabel("Hours")
ax.set_ylabel("Energy Units (MW)")
ax.legend()

st.pyplot(fig)

# ==========================
# FEATURE IMPORTANCE
# ==========================

st.subheader("Feature Importance")

importance = model.feature_importances_

imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

fig2, ax2 = plt.subplots()

sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax2)

st.pyplot(fig2)

# ==========================
# PREDICTION UI
# ==========================

st.subheader("Predict Future Energy Demand")

hour = st.slider("Hour of Day",0,23,12)
day = st.slider("Day of Week (0=Monday)",0,6,3)
month = st.slider("Month",1,12,6)
weekend = st.selectbox("Weekend?", [0,1])
night = st.selectbox("Night?", [0,1])

input_data = np.array([[hour,day,month,weekend,night]])

if st.button("Predict Energy Demand"):

    prediction = model.predict(input_data)

    st.success(f"Predicted Energy Demand: {round(prediction[0],2)} MW")

# ==========================
# ANOMALY DETECTION
# ==========================

st.subheader("Energy Waste Detection")

results["Error"] = results["Actual"] - results["Predicted"]

threshold = results["Predicted"] * 0.15

anomalies = results[results["Error"] > threshold]

st.write("Number of anomaly hours detected:", len(anomalies))

# ==========================
# SUSTAINABILITY REPORT
# ==========================

st.subheader("Sustainability Report")

total_wasted_mw = anomalies["Error"].sum()

co2_saved_kg = total_wasted_mw * 1000 * 0.19
co2_saved_tonnes = co2_saved_kg / 1000

col1, col2, col3 = st.columns(3)

col1.metric("Energy Waste (MWh)", round(total_wasted_mw,2))
col2.metric("CO₂ Reduction (tonnes)", round(co2_saved_tonnes,2))
col3.metric("Equivalent Trees Planted", int(co2_saved_tonnes*45))

st.success("SMARTGRID-AI helps improve energy efficiency and sustainability 🌍")