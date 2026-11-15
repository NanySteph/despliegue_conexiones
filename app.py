"""
Clasificador de conexiones de red (Normal / Sospechosa)
4 kernels SVM + Árbol de Decisión — Desarrollado por: Stephanie Machado
Ejecutar local:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay)

AUTORA = "Stephanie Machado"
COLORES = ["#0891b2", "#f97316"]
SEMILLA = 240
VARIABLES = ["trafico_mbps", "latencia_ms"]

st.set_page_config(page_title=f"Kernels SVM y Árbol — {AUTORA}", layout="wide")
st.title("Clasificación de conexiones de red")
st.markdown(f"**Elaborado por: {AUTORA}**")
st.caption("Comparación de cuatro kernels SVM y un árbol de decisión para clasificar "
           "conexiones como normales (0) o sospechosas (1).")


def firma(fig):
    fig.text(0.99, 0.01, f"Elaborado por {AUTORA}", ha="right", fontsize=8, color="gray")


@st.cache_data
def leer(archivo):
    return pd.read_csv(archivo)


# ---------- datos ----------
subido = st.sidebar.file_uploader("Cargar CSV", type="csv")
try:
    datos = leer(subido if subido else "conexiones_red.csv")
except FileNotFoundError:
    st.info("Sube el archivo `conexiones_red.csv` en la barra lateral.")
    st.stop()

faltan = [c for c in VARIABLES + ["clase"] if c not in datos.columns]
if faltan:
    st.error(f"Faltan columnas: {faltan}")
    st.stop()


@st.cache_resource
def entrenar(df):
    X, y = df[VARIABLES], df["clase"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEMILLA, stratify=y)
    modelos = {
        "SVM Lineal": make_pipeline(StandardScaler(), SVC(kernel="linear", C=1)),
        "SVM Polinomial": make_pipeline(StandardScaler(), SVC(kernel="poly", degree=3, C=1)),
        "SVM RBF": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1)),
        "SVM Sigmoide": make_pipeline(StandardScaler(), SVC(kernel="sigmoid", C=1)),
        "Árbol de decisión": make_pipeline(DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=8, random_state=SEMILLA)),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    filas, res = [], []
    for n, m in modelos.items():
        r = cross_validate(m, Xtr, ytr, cv=cv, scoring=["accuracy", "f1"])
        filas.append({"Modelo": n, "Accuracy CV": r["test_accuracy"].mean(),
                      "F1 CV": r["test_f1"].mean(), "Desv. F1": r["test_f1"].std()})
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        res.append({"Modelo": n, "Accuracy": accuracy_score(yte, p),
                    "Precisión": precision_score(yte, p, zero_division=0),
                    "Recall": recall_score(yte, p, zero_division=0),
                    "F1": f1_score(yte, p, zero_division=0)})
    tcv = pd.DataFrame(filas).set_index("Modelo").sort_values("F1 CV", ascending=False)
    ttest = pd.DataFrame(res).set_index("Modelo")
    return modelos, Xtr, Xte, ytr, yte, tcv, ttest


modelos, Xtr, Xte, ytr, yte, tabla_cv, tabla_test = entrenar(datos)
seleccionado = tabla_cv.index[0]

# ---------- predicción (barra lateral) ----------
st.sidebar.header("Predecir nueva conexión")
traf = st.sidebar.number_input("Tráfico (Mbps)", value=100.0)
lat = st.sidebar.number_input("Latencia (ms)", value=100.0)
nueva = pd.DataFrame({"trafico_mbps": [traf], "latencia_ms": [lat]})
for n, m in modelos.items():
    p = m.predict(nueva)[0]
    txt = "🟠 Sospechosa" if p == 1 else "🔵 Normal"
    st.sidebar.write(f"**{n}**{' ⭐' if n == seleccionado else ''}: {txt}")
st.sidebar.caption("⭐ = modelo seleccionado por validación cruzada")
st.sidebar.markdown(f"---\n© {AUTORA}")

# ---------- pestañas ----------
t1, t2, t3, t4, t5 = st.tabs(["Datos", "Validación", "Fronteras SVM", "Árbol", "Evaluación"])

with t1:
    st.write(f"{len(datos)} filas — Entrenamiento: {len(Xtr)} / Prueba: {len(Xte)}")
    st.dataframe(datos.head(10))
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, et in [(0, "Normales (0)"), (1, "Sospechosas (1)")]:
        d = datos[datos["clase"] == c]
        ax.scatter(d["trafico_mbps"], d["latencia_ms"], c=COLORES[c], label=et, alpha=.8, edgecolors="w")
    ax.set(title="Tráfico vs Latencia", xlabel="Tráfico (Mbps)", ylabel="Latencia (ms)")
    ax.legend(); firma(fig); st.pyplot(fig)

with t2:
    st.subheader("Validación cruzada estratificada (5 particiones)")
    st.dataframe(tabla_cv.round(3))
    st.success(f"Modelo seleccionado por F1: **{seleccionado}**")

with t3:
    xx, yy = np.meshgrid(np.linspace(Xtr.iloc[:, 0].min() - 4, Xtr.iloc[:, 0].max() + 4, 200),
                         np.linspace(Xtr.iloc[:, 1].min() - 4, Xtr.iloc[:, 1].max() + 4, 200))
    rej = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=VARIABLES)
    fig, ejes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, (n, m) in zip(ejes.ravel(), list(modelos.items())[:4]):
        z = m.decision_function(rej).reshape(xx.shape)
        ax.contourf(xx, yy, (z > 0).astype(int), levels=[-.5, .5, 1.5], colors=COLORES, alpha=.15)
        ax.contour(xx, yy, z, levels=[-1, 0, 1], colors=["#64748b", "#102a43", "#64748b"],
                   linestyles=["--", "-", "--"])
        for c in (0, 1):
            pts = Xtr[ytr == c]
            ax.scatter(pts.iloc[:, 0], pts.iloc[:, 1], c=COLORES[c], s=15, edgecolors="white", linewidths=.3)
        sv = Xtr.iloc[m.named_steps["svc"].support_]
        ax.scatter(sv.iloc[:, 0], sv.iloc[:, 1], s=60, facecolors="none", edgecolors="#102a43", linewidths=.6)
        ax.set_title(f"{n} | {len(sv)} vectores de soporte")
        ax.set_xlabel("Tráfico (Mbps)"); ax.set_ylabel("Latencia (ms)")
    fig.suptitle("Una misma base de datos, cuatro formas de separar")
    firma(fig); fig.tight_layout(); st.pyplot(fig)

with t4:
    arbol = modelos["Árbol de decisión"].named_steps["decisiontreeclassifier"]
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(arbol, feature_names=VARIABLES, class_names=["Normal", "Sospechosa"],
              filled=True, rounded=True, precision=2, fontsize=9, ax=ax)
    ax.set_title("Árbol de decisión · profundidad máxima 4")
    firma(fig); st.pyplot(fig)

with t5:
    st.dataframe(tabla_test.round(3))
    fig, ejes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, (n, m) in zip(ejes, modelos.items()):
        ConfusionMatrixDisplay.from_predictions(yte, m.predict(Xte), display_labels=["Normal", "Sosp."],
                                                colorbar=False, cmap="Blues", ax=ax)
        ax.set_title(n, fontsize=10); ax.grid(False)
    firma(fig); fig.tight_layout(); st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    tabla_test[["Accuracy", "F1"]].plot.bar(ax=ax, color=COLORES, ylim=(0, 1.08), rot=15)
    ax.set(title="Desempeño en prueba", ylabel="Puntuación")
    firma(fig); fig.tight_layout(); st.pyplot(fig)

st.markdown("---")
st.caption(f"Desarrollado por {AUTORA}")
