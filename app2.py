import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Riesgo de Crédito", page_icon="💳", layout="centered")

# ── Cargar artefactos ──────────────────────────────────────────────────────────
@st.cache_resource
def cargar_artefactos():
    encoders = joblib.load("label_encoders.joblib")
    pca      = joblib.load("modelo_pca.joblib")
    scaler   = joblib.load("minmax_scaler.joblib")
    modelo   = load_model("modelo_red_neuronal.keras")
    return encoders, pca, scaler, modelo

encoders, pca, scaler, modelo = cargar_artefactos()

# ── Constantes ─────────────────────────────────────────────────────────────────
# Pipeline confirmado:
# [Num_Credit_Card, Interest_Rate, Delay_from_due_date,
#  Num_Credit_Inquiries, Payment_of_Min_Amount]
# → PCA(3) → MinMaxScaler(3) → Red Neuronal

LABEL_MAP = {0: "⚠️ Poor", 1: "🟡 Standard", 2: "✅ Good"}
COLOR_MAP  = {0: "#FF4B4B", 1: "#FFA500",    2: "#21C354"}

RECOMENDACIONES = {
    0: {
        "icono": "🚨", "titulo": "Alto riesgo — Acción urgente requerida", "color": "#FF4B4B",
        "puntos": [
            "Reducir inmediatamente los días de retraso en pagos.",
            "Evitar nuevas solicitudes de crédito por al menos 6 meses.",
            "Pagar las deudas priorizando las de mayor tasa de interés.",
            "Cubrir al menos el monto mínimo mensual sin falta.",
            "Considerar asesoría financiera profesional.",
        ],
    },
    1: {
        "icono": "⚠️", "titulo": "Riesgo moderado — Hay margen de mejora", "color": "#FFA500",
        "puntos": [
            "Mantener los pagos al día para mejorar la categoría.",
            "Reducir la deuda pendiente al menos un 20% en los próximos 3 meses.",
            "Limitar las consultas de crédito a las estrictamente necesarias.",
            "Revisar si el número de tarjetas de crédito es manejable.",
        ],
    },
    2: {
        "icono": "✅", "titulo": "Perfil crediticio saludable", "color": "#21C354",
        "puntos": [
            "Mantener el buen comportamiento de pago.",
            "Aprovechar la buena calificación para negociar mejores tasas.",
            "Mantener la deuda por debajo del 30% del límite disponible.",
            "Evitar abrir muchas cuentas nuevas al mismo tiempo.",
        ],
    },
}

# ── Interfaz ───────────────────────────────────────────────────────────────────
st.title("💳 Predictor de Riesgo de Crédito")
st.markdown("Ingresa los datos del cliente para predecir su **Credit Score**.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Datos financieros")
    num_credit_card      = st.number_input("N° Tarjetas de crédito",   min_value=0.0, max_value=15.0, value=5.0,  step=0.5)
    interest_rate        = st.number_input("Tasa de interés (%)",      min_value=1,   max_value=34,   value=14,   step=1)
    delay_from_due_date  = st.number_input("Días de retraso en pagos", min_value=0.0, max_value=70.0, value=10.0, step=0.5)
    num_credit_inquiries = st.number_input("N° Consultas de crédito",  min_value=0.0, max_value=17.0, value=5.0,  step=0.5)

with col2:
    st.subheader("🏷️ Comportamiento de pago")
    payment_of_min_amount = st.selectbox("¿Paga el monto mínimo?", options=["Yes", "No", "NM"], index=0)
    st.markdown("---")
    st.info("El modelo usa estas **5 variables clave** identificadas durante el entrenamiento con SelectKBest.")

st.divider()

# ── Predicción ─────────────────────────────────────────────────────────────────
if st.button("🔍 Predecir riesgo de crédito", use_container_width=True, type="primary"):

    # 1. Codificar Payment_of_Min_Amount con LabelEncoder
    le_payment = encoders["Payment_of_Min_Amount"]
    try:
        payment_enc = le_payment.transform([payment_of_min_amount])[0]
    except ValueError:
        payment_enc = le_payment.transform(["Yes"])[0]

    # 2. Construir vector con las 5 features en el orden exacto del entrenamiento
    X_input = np.array([[
        num_credit_card,
        interest_rate,
        delay_from_due_date,
        num_credit_inquiries,
        payment_enc,
    ]])

    # 3. PCA: 5 features → 3 componentes
    X_pca = pca.transform(X_input)

    # 4. MinMaxScaler sobre los 3 componentes
    X_scaled = scaler.transform(X_pca)

    # 5. Predecir con la red neuronal
    probas    = modelo.predict(X_scaled, verbose=0)[0]
    clase     = int(np.argmax(probas))
    confianza = float(probas[clase]) * 100

    # ── Resultado ──────────────────────────────────────────────────────────────
    st.subheader("📋 Resultado de la predicción")
    color = COLOR_MAP[clase]
    label = LABEL_MAP[clase]

    st.markdown(
        f"""<div style="background-color:{color}22; border-left:5px solid {color};
                padding:16px; border-radius:8px; margin-bottom:16px;">
            <h2 style="color:{color}; margin:0;">{label}</h2>
            <p style="margin:4px 0 0 0; color:#555;">Confianza: <strong>{confianza:.1f}%</strong></p>
        </div>""",
        unsafe_allow_html=True,
    )

    # Barras de probabilidad
    st.markdown("#### 📊 Probabilidades")
    for i in range(3):
        es_pred = clase == i
        st.markdown(
            f"<div style='font-weight:{'700' if es_pred else '400'}'>"
            f"{LABEL_MAP[i]}{' ◀ predicción' if es_pred else ''}</div>",
            unsafe_allow_html=True,
        )
        st.progress(float(probas[i]), text=f"{probas[i]*100:.2f}%")

    with st.expander("🔢 Ver tabla de probabilidades"):
        prob_df = pd.DataFrame({
            "Clase":        [LABEL_MAP[i] for i in range(3)],
            "Probabilidad": [f"{p*100:.2f}%" for p in probas],
        })
        st.dataframe(prob_df, hide_index=True, use_container_width=True)

    # Recomendaciones
    st.markdown("#### 💡 Recomendaciones")
    rec = RECOMENDACIONES[clase]
    st.markdown(
        f"<div style='background-color:{rec['color']}22; border-left:6px solid {rec['color']}; "
        f"padding:14px 18px; border-radius:8px; margin-bottom:14px'>"
        f"<span style='font-size:1.1rem; font-weight:700; color:{rec['color']}'>"
        f"{rec['icono']}  {rec['titulo']}</span></div>",
        unsafe_allow_html=True,
    )
    for i, punto in enumerate(rec["puntos"], 1):
        st.markdown(f"**{i}.** {punto}")

st.divider()
st.caption("Pipeline: LabelEncoder → SelectKBest(5) → PCA(3) → MinMaxScaler → Red Neuronal")
