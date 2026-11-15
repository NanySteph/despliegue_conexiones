"""
Detector de conexiones sospechosas
Desarrollado por: Stephanie Machado
Ejecutar local:  streamlit run app.py
"""
import pandas as pd
import streamlit as st
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

AUTORA = "Stephanie Machado"
SEMILLA = 240
VARIABLES = ["trafico_mbps", "latencia_ms"]

st.set_page_config(page_title="Detector de conexiones", page_icon="📡", layout="centered")


@st.cache_resource
def entrenar():
    datos = pd.read_csv("conexiones_red.csv")
    X, y = datos[VARIABLES], datos["clase"]
    candidatos = {
        "SVM Lineal": make_pipeline(StandardScaler(), SVC(kernel="linear", C=1)),
        "SVM Polinomial": make_pipeline(StandardScaler(), SVC(kernel="poly", degree=3, C=1)),
        "SVM RBF": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1)),
        "SVM Sigmoide": make_pipeline(StandardScaler(), SVC(kernel="sigmoid", C=1)),
        "Árbol de decisión": DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, random_state=SEMILLA),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    puntajes = {n: cross_val_score(m, X, y, cv=cv, scoring="f1").mean() for n, m in candidatos.items()}
    mejor = max(puntajes, key=puntajes.get)
    return candidatos[mejor].fit(X, y), mejor


st.title("📡 Detector de conexiones de red")
st.caption(f"Elaborado por: **{AUTORA}**")
st.markdown("💻 Equipo ── 📶 Red ── 🛡️ Detector ── 🖥️ Servidor")
st.write("Ingresa los datos de la conexión para saber si es **normal** o **sospechosa**.")

try:
    modelo, nombre = entrenar()
except FileNotFoundError:
    st.error("No se encontró `conexiones_red.csv` junto a app.py.")
    st.stop()

with st.container(border=True):
    c1, c2 = st.columns(2)
    trafico = c1.number_input("📶 Tráfico (Mbps)", min_value=0.0, value=50.0, step=1.0)
    latencia = c2.number_input("⏱️ Latencia (ms)", min_value=0.0, value=70.0, step=1.0)
    revisar = st.button("Revisar conexión", type="primary", width="stretch")

if revisar:
    pred = modelo.predict(pd.DataFrame({"trafico_mbps": [trafico], "latencia_ms": [latencia]}))[0]
    if pred == 1:
        st.error("🚨 **Conexión SOSPECHOSA** — se recomienda bloquearla y revisarla.")
    else:
        st.success("✅ **Conexión NORMAL** — el tráfico puede continuar.")

st.divider()
st.caption(f"Modelo usado: {nombre} · © {AUTORA}")
