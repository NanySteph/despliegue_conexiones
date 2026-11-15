"""
PacketScope · Analizador de tráfico de red (estilo Wireshark)
Clasifica conexiones como Normales o Sospechosas a partir del tráfico y la latencia.
Desarrollado por: Stephanie Machado
Ejecutar local:  streamlit run app.py
"""
import random
from datetime import datetime, timedelta

import numpy as np
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
PUERTOS = {"HTTPS": 443, "HTTP": 80, "SSH": 22, "DNS": 53, "FTP": 21, "RDP": 3389, "SMTP": 25}

st.set_page_config(page_title="PacketScope · Analizador de red", page_icon="📡", layout="wide")

st.markdown("""
<style>
.stApp { background:#f4f7fb; }
.barra { background:linear-gradient(90deg,#1e3a8a,#2563eb); color:white; padding:14px 22px;
  border-radius:10px; display:flex; justify-content:space-between; align-items:center; }
.barra h1 { color:white !important; margin:0; font-size:28px; }
.barra .sub { font-size:13px; opacity:.85; }
.autora { background:#fbbf24; color:#1e293b; padding:6px 12px; border-radius:20px; font-weight:700; font-size:13px; }
.capa { border-left:4px solid #2563eb; background:white; padding:8px 14px; margin:4px 0;
  border-radius:6px; font-family:monospace; font-size:13px; box-shadow:0 1px 3px rgba(0,0,0,.06); }
.capa b { color:#1e3a8a; }
.medidor { background:white; border-radius:12px; padding:14px; text-align:center;
  box-shadow:0 2px 8px rgba(0,0,0,.07); }
.medidor .v { font-size:30px; font-weight:800; }
.medidor .l { font-size:12px; color:#64748b; text-transform:uppercase; letter-spacing:1px; }
.barra-n { height:8px; border-radius:4px; background:#e2e8f0; margin-top:8px; overflow:hidden; }
.veredicto { border-radius:12px; padding:20px; text-align:center; font-size:24px; font-weight:800; }
.sos { background:#fee2e2; color:#991b1b; border:2px solid #ef4444; }
.nor { background:#dcfce7; color:#166534; border:2px solid #22c55e; }
</style>""", unsafe_allow_html=True)


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
    return candidatos[mejor].fit(X, y), mejor, puntajes[mejor], datos


def medidor(valor, maximo, etiqueta, unidad, color):
    pct = min(valor / maximo, 1) * 100
    return (f'<div class="medidor"><div class="l">{etiqueta}</div>'
            f'<div class="v" style="color:{color}">{valor:.1f} <span style="font-size:14px">{unidad}</span></div>'
            f'<div class="barra-n"><div style="width:{pct}%;height:100%;background:{color}"></div></div></div>')


def ip_aleatoria(privada=True):
    return f"192.168.{random.randint(0, 5)}.{random.randint(2, 254)}" if privada else \
           f"{random.randint(11, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


# ---------------- encabezado ----------------
st.markdown(f"""
<div class="barra">
  <div><h1>📡 PacketScope</h1><div class="sub">Analizador de tráfico y detección de conexiones sospechosas</div></div>
  <div class="autora">👩‍💻 Elaborado por: {AUTORA}</div>
</div>""", unsafe_allow_html=True)
st.write("")

try:
    modelo, nombre, f1, datos = entrenar()
except FileNotFoundError:
    st.error("No se encontró `conexiones_red.csv` junto a app.py.")
    st.stop()

tab1, tab2 = st.tabs(["🔍 Inspeccionar conexión", "🌐 Escanear la red"])

# ======================= PESTAÑA 1 =======================
with tab1:
    c_in, c_out = st.columns([1, 1.5])
    with c_in:
        st.subheader("Datos de la conexión")
        servicio = st.selectbox("Servicio", list(PUERTOS))
        ip_src = st.text_input("IP origen", "192.168.1.25")
        ip_dst = st.text_input("IP destino", "172.217.3.110")
        trafico = st.number_input("📶 Tráfico (Mbps)", 0.0, 1000.0, 50.0, 1.0)
        latencia = st.number_input("⏱️ Latencia (ms)", 0.0, 1000.0, 70.0, 1.0)
        boton = st.button("🔍 Inspeccionar", type="primary", width="stretch")

    with c_out:
        st.subheader("Modelo OSI de la conexión")
        st.markdown(f"""
<div class="capa"><b>L7 Aplicación</b> · {servicio}</div>
<div class="capa"><b>L4 Transporte</b> · {'UDP' if servicio == 'DNS' else 'TCP'} · puerto destino {PUERTOS[servicio]}</div>
<div class="capa"><b>L3 Red</b> · IPv4 {ip_src} → {ip_dst}</div>
<div class="capa"><b>Métricas</b> · ancho de banda {trafico:.1f} Mbps · RTT {latencia:.1f} ms</div>
""", unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.markdown(medidor(trafico, 200, "Tráfico", "Mbps", "#2563eb"), unsafe_allow_html=True)
        m2.markdown(medidor(latencia, 250, "Latencia", "ms", "#f59e0b"), unsafe_allow_html=True)
        st.write("")

        if boton:
            pred = int(modelo.predict(pd.DataFrame({"trafico_mbps": [trafico], "latencia_ms": [latencia]}))[0])
            if pred == 1:
                st.markdown(f'<div class="veredicto sos">⚠️ CONEXIÓN SOSPECHOSA<br>'
                            f'<span style="font-size:14px;font-weight:400">{ip_src} → {ip_dst}:{PUERTOS[servicio]} '
                            f'· Acción sugerida: bloquear y revisar</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="veredicto nor">✅ CONEXIÓN NORMAL<br>'
                            f'<span style="font-size:14px;font-weight:400">{ip_src} → {ip_dst}:{PUERTOS[servicio]} '
                            f'· Tráfico permitido</span></div>', unsafe_allow_html=True)

            # Simulación de 30 s de tráfico alrededor de los valores ingresados
            rng = np.random.default_rng()
            t = [(datetime.now() - timedelta(seconds=30 - i)).strftime("%H:%M:%S") for i in range(30)]
            serie = pd.DataFrame({"Tráfico (Mbps)": np.clip(rng.normal(trafico, trafico * .08 + 1, 30), 0, None),
                                  "Latencia (ms)": np.clip(rng.normal(latencia, latencia * .08 + 1, 30), 0, None)},
                                 index=t)
            st.caption("Monitoreo simulado de los últimos 30 segundos")
            st.line_chart(serie, color=["#2563eb", "#f59e0b"], height=220)

# ======================= PESTAÑA 2 =======================
with tab2:
    st.subheader("Escaneo de conexiones activas")
    st.write("Toma una muestra de conexiones del registro de tráfico y las clasifica todas.")
    n = st.slider("Número de conexiones a escanear", 10, min(200, len(datos)), 25)
    if st.button("🌐 Iniciar escaneo", type="primary"):
        muestra = datos.sample(n).reset_index(drop=True)
        pred = modelo.predict(muestra[VARIABLES])
        servicios = random.choices(list(PUERTOS), k=n)
        tabla = pd.DataFrame({
            "No.": range(1, n + 1),
            "Origen": [ip_aleatoria() for _ in range(n)],
            "Destino": [ip_aleatoria(False) for _ in range(n)],
            "Servicio": servicios,
            "Puerto": [PUERTOS[s] for s in servicios],
            "Tráfico (Mbps)": muestra["trafico_mbps"].round(1),
            "Latencia (ms)": muestra["latencia_ms"].round(1),
            "Estado": np.where(pred == 1, "⚠️ Sospechosa", "✅ Normal"),
        })
        sos = int((pred == 1).sum())
        k1, k2, k3 = st.columns(3)
        k1.metric("Conexiones escaneadas", n)
        k2.metric("✅ Normales", n - sos)
        k3.metric("⚠️ Sospechosas", sos, delta=f"{sos / n:.0%} del total", delta_color="inverse")

        def color_fila(fila):
            c = "background-color:#fee2e2" if "Sospechosa" in fila["Estado"] else ""
            return [c] * len(fila)

        st.dataframe(tabla.style.apply(color_fila, axis=1), hide_index=True, width="stretch")
        st.caption("Mapa de conexiones: azul = normal, rojo = sospechosa")
        mapa = tabla[["Tráfico (Mbps)", "Latencia (ms)"]].copy()
        mapa["color"] = np.where(pred == 1, "#ef4444", "#2563eb")
        st.scatter_chart(mapa, x="Tráfico (Mbps)", y="Latencia (ms)", color="color", height=300)

st.divider()
st.caption(f"Motor de clasificación: {nombre} (F1 validación cruzada = {f1:.3f}) · "
           f"PacketScope © Desarrollado por {AUTORA}")
