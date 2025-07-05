import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
@st.cache_data
def load_data():
    # Dynamically find path to the CSV (works on Thonny and Streamlit Cloud)
    file_path = os.path.join(os.path.dirname(__file__), "HousingData.csv")
    df = pd.read_csv(file_path)
    df.replace("NA", np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[df["MEDV"].notna()]
    return df

df = load_data()

st.title("🏠 Boston House Price Predictor")

# Preprocess
imputer = SimpleImputer(strategy="mean")
X = df.drop("MEDV", axis=1)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
y = df["MEDV"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.markdown("### 📥 Enter Housing Features")

# Input section with explanations
def feature_input():
    values = {}
    values["CRIM"] = st.number_input("CRIM", value=3.62)
    st.caption("Per capita crime rate by town")

    values["ZN"] = st.number_input("ZN", value=11.22)
    st.caption("Proportion of residential land zoned for large lots")

    values["INDUS"] = st.number_input("INDUS", value=11.09)
    st.caption("Proportion of non-retail business acres per town")

    values["CHAS"] = st.number_input("CHAS", value=0.0)
    st.caption("1 if tract bounds Charles River, else 0")

    values["NOX"] = st.number_input("NOX", value=0.6)
    st.caption("Nitric oxide concentration (parts per 10 million)")

    values["RM"] = st.number_input("RM", value=6.3)
    st.caption("Average number of rooms per dwelling")

    values["AGE"] = st.number_input("AGE", value=68.5)
    st.caption("Proportion of units built before 1940")

    values["DIS"] = st.number_input("DIS", value=3.8)
    st.caption("Weighted distance to employment centers")

    values["RAD"] = st.number_input("RAD", value=9.5)
    st.caption("Accessibility to radial highways")

    values["TAX"] = st.number_input("TAX", value=408.0)
    st.caption("Property-tax rate per $10,000")

    values["PTRATIO"] = st.number_input("PTRATIO", value=18.5)
    st.caption("Pupil–teacher ratio by town")

    values["B"] = st.number_input("B", value=356.7)
    st.caption("1000(Bk - 0.63)^2, where Bk = % of Black residents")

    values["LSTAT"] = st.number_input("LSTAT", value=12.5)
    st.caption("Percentage of lower status population")

    return pd.DataFrame([values])

input_df = feature_input()

# Prediction
input_df = pd.DataFrame(imputer.transform(input_df), columns=X.columns)
prediction = model.predict(input_df)[0]

# Show result
st.markdown("### 🏡 Predicted House Price")
st.success(f"${prediction * 1000:,.2f} (in USD)")

# Evaluation shown in small font
st.markdown("---")
st.caption(f"📉 Mean Squared Error: {mse:.2f}")
st.caption(f"✅ R² Score: {r2 * 100:.2f}%")

