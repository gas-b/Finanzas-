import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("A01615309_Actividad2_Registro-1.csv")

df["Fecha (dd/mm/aa)"] = pd.to_datetime(df["Fecha (dd/mm/aa)"], errors="coerce")

df["Año"] = df["Fecha (dd/mm/aa)"].dt.year
df["Mes"] = df["Fecha (dd/mm/aa)"].dt.month
df["Dia"] = df["Fecha (dd/mm/aa)"].dt.day
df["Dia_semana"] = df["Fecha (dd/mm/aa)"].dt.dayofweek

df["Tiempo invertido"] = (
    df["Tiempo invertido"]
    .astype(str)
    .str.replace(" min", "", regex=False)
    .astype(float)
)

df = pd.get_dummies(df, columns=["Nombre actividad", "Tipo", "Momento"], drop_first=True)

df = df.drop(["Número", "Fecha (dd/mm/aa)"], axis=1)

X = df.drop("Costo", axis=1)
y = df["Costo"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = DecisionTreeRegressor(max_depth=5, random_state=42)
modelo.fit(X_train, y_train)

st.title("📊 Predicción de Costos de Actividades")
st.image("Finanzas🤑.jpg", caption="Ingresa datos y obtén el costo estimado.")

presupuesto = st.number_input("Presupuesto asignado", min_value=0.0)
tiempo = st.number_input("Tiempo invertido (min)", min_value=0.0)
personas = st.number_input("Número de personas", min_value=1.0)

actividad = st.selectbox("Actividad", X.filter(like="Nombre").columns)
tipo = st.selectbox("Tipo", X.filter(like="Tipo").columns)
momento = st.selectbox("Momento del día", X.filter(like="Momento").columns)

entrada = np.zeros(len(X.columns))

entrada[X.columns.get_loc("Presupuesto")] = presupuesto
entrada[X.columns.get_loc("Tiempo invertido")] = tiempo
entrada[X.columns.get_loc("No. de personas")] = personas

entrada[X.columns.get_loc(actividad)] = 1
entrada[X.columns.get_loc(tipo)] = 1
entrada[X.columns.get_loc(momento)] = 1

if st.button("Predecir costo"):
    pred = modelo.predict([entrada])[0]
    st.success(f"💸 Costo estimado: **${pred:.2f}**")

st.header("⚖ Equilibrio entre ingresos y gastos")

ingreso = st.number_input("¿Cuánto ganas al mes?", min_value=0.0)
reinversion = st.number_input("Reinversiones o ingresos extras", min_value=0.0)

total_ingreso = ingreso + reinversion
costo_mensual = df["Costo"].mean() * 30

if st.button("Evaluar equilibrio"):
    if total_ingreso >= costo_mensual:
        st.success(f"✔ Estás en equilibrio (superávit de ${total_ingreso - costo_mensual:.2f})")
    else:
        deficit = costo_mensual - total_ingreso
        st.error(f"❌ Estás en déficit de ${deficit:.2f}")

        reduccion = (deficit / costo_mensual) * 100
        st.warning(f"Debes reducir gastos un {reduccion:.1f}% o aumentar ingresos en ${deficit:.2f}")
