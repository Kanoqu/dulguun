import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# DATASET
# -----------------------------
data = {
    "year": list(range(2014, 2025)),

    "gdp": [
        22227054,22894780,23931342,28010710,32582629,
        37839225,37883041,44702733,54877815,80208373,80208373
    ],

    "inflation": [12.6,5.9,0.6,6.7,8.1,7.3,3.7,7.3,13.2,10.4,9],

    "budget": [7.1,7.2,8.6,9.5,9.6,12,14.4,15.7,18,22.5,30.3],

    "fx": [1800,1970,2140,2440,2460,2660,2850,2900,3300,3450,3400],

    "rate": [12.5,13,12.6,13,10,11,9,6,10,13,11.7]
}

df = pd.DataFrame(data)

# -----------------------------
# MODEL
# -----------------------------
X = df[["gdp","inflation","budget","fx"]]
y = df["rate"]

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, y)

# -----------------------------
# UI
# -----------------------------
st.title("🏦 Монгол Улсын Бодлогын Хүү AI System")

st.subheader("📊 Өгөгдөл")
st.dataframe(df)

# -----------------------------
# INPUT
# -----------------------------
gdp = st.number_input("ДНБ", value=80000000)
infl = st.number_input("Инфляци", value=9.0)
budget = st.number_input("Төсөв", value=30.0)
fx = st.number_input("USD/MNT ханш", value=3400)

# -----------------------------
# PREDICT
# -----------------------------
if st.button("🔮 Таамаглах"):
    pred = model.predict([[gdp, infl, budget, fx]])[0]
    st.success(f"🏦 Бодлогын хүү таамаг: {round(pred,2)} %")