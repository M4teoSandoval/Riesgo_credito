import streamlit as st
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model

# ──────────────────────────────────────────────
# Configuración de la página
# ──────────────────────────────────────────────
st.set_page_config(page_title="Riesgo de Crédito", page_icon="💳", layout="centered")


# ──────────────────────────────────────────────
# Carga de artefactos (se cachea para no recargar)
# ──────────────────────────────────────────────
@st.cache_resource
def cargar_artefactos():
    encoders = joblib.load("label_encoders.joblib")  # dict con LabelEncoders
    scaler = joblib.load("minmax_scaler.joblib")  # MinMaxScaler
    pca = joblib.load("modelo_pca.joblib")  # PCA (3 componentes)
    modelo = load_model("modelo_red_neuronal.keras")  # Red neuronal
    return encoders, scaler, pca, modelo


encoders, scaler, pca, modelo = cargar_artefactos()

# ──────────────────────────────────────────────
# Constantes del pipeline
# ──────────────────────────────────────────────
# Orden exacto de features ANTES de SelectKBest
TODAS_LAS_FEATURES = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Credit_Mix",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Age",
    "Payment_of_Min_Amount",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
]

# Las 5 features seleccionadas por SelectKBest (en el mismo orden que X.columns)
FEATURES_SELECCIONADAS = [
    "Num_Credit_Card",
    "Interest_Rate",
    "Delay_from_due_date",
    "Num_Credit_Inquiries",
    "Payment_of_Min_Amount",
]

# Columnas codificadas con LabelEncoder
COLS_CATEGORICAS = ["Credit_Mix", "Payment_of_Min_Amount"]

CLASES = {0: "⚠️ Poor", 1: "🟡 Standard", 2: "✅ Good"}
COLORES = {0: "#FF4B4B", 1: "#FFA500", 2: "#21C354"}

# ──────────────────────────────────────────────
# Interfaz
# ──────────────────────────────────────────────
st.title("💳 Predictor de Riesgo de Crédito")
st.markdown("Ingresa los datos del cliente para predecir su **Credit Score**.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Información financiera")
    num_credit_card = st.number_input(
        "Número de tarjetas de crédito",
        min_value=0.0,
        max_value=15.0,
        value=5.0,
        step=0.5,
    )
    interest_rate = st.number_input(
        "Tasa de interés (%)", min_value=1, max_value=34, value=14, step=1
    )
    delay_from_due_date = st.number_input(
        "Días de retraso en pagos", min_value=-5.0, max_value=70.0, value=20.0, step=0.5
    )
    num_credit_inquiries = st.number_input(
        "Número de consultas de crédito",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
    )

with col2:
    st.subheader("🏷️ Comportamiento de pago")
    payment_of_min_amount = st.selectbox(
        "¿Paga el monto mínimo?", options=["Yes", "No", "NM"], index=0
    )
    # Información adicional (no entra al modelo pero contextualiza)
    st.markdown("---")
    st.info(
        "Solo se usan las **5 variables más importantes** identificadas por el modelo durante el entrenamiento."
    )

st.divider()

# ──────────────────────────────────────────────
# Predicción
# ──────────────────────────────────────────────
if st.button("🔍 Predecir riesgo de crédito", use_container_width=True, type="primary"):
    # 1. Codificar Payment_of_Min_Amount
    le_payment = encoders["Payment_of_Min_Amount"]
    try:
        payment_encoded = le_payment.transform([payment_of_min_amount])[0]
    except ValueError:
        # Si "NM" no estaba en el entrenamiento, usar el más frecuente
        payment_encoded = le_payment.transform(["Yes"])[0]

    # 2. Construir array con las 5 features seleccionadas
    #    Orden: Num_Credit_Card, Interest_Rate, Delay_from_due_date,
    #           Num_Credit_Inquiries, Payment_of_Min_Amount
    X_input = np.array(
        [
            [
                num_credit_card,
                interest_rate,
                delay_from_due_date,
                num_credit_inquiries,
                payment_encoded,
            ]
        ]
    )

    # 3. Escalar con MinMaxScaler
    X_scaled = scaler.transform(X_input)

    # 4. Reducir con PCA
    X_pca = pca.transform(X_scaled)

    # 5. Predecir
    probas = modelo.predict(X_pca, verbose=0)[0]
    clase = int(np.argmax(probas))
    confianza = float(probas[clase]) * 100

    # ── Resultado ──
    st.subheader("📋 Resultado de la predicción")

    color = COLORES[clase]
    label = CLASES[clase]

    st.markdown(
        f"""
        <div style="background-color:{color}22; border-left: 5px solid {color};
                    padding: 16px; border-radius: 8px; margin-bottom: 16px;">
            <h2 style="color:{color}; margin:0;">{label}</h2>
            <p style="margin:4px 0 0 0; color:#555;">Confianza: <strong>{confianza:.1f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Probabilidades por clase
    st.markdown("**Distribución de probabilidades:**")
    prob_df = pd.DataFrame(
        {
            "Clase": [CLASES[i] for i in range(3)],
            "Probabilidad": [f"{p * 100:.1f}%" for p in probas],
        }
    )
    st.dataframe(prob_df, hide_index=True, use_container_width=True)

    # Interpretación
    st.markdown("---")
    if clase == 0:
        st.error(
            "**Poor** → Alto riesgo crediticio. Se recomienda mayor análisis antes de aprobar crédito."
        )
    elif clase == 1:
        st.warning(
            "**Standard** → Riesgo moderado. El cliente puede ser elegible con condiciones estándar."
        )
    else:
        st.success(
            "**Good** → Bajo riesgo crediticio. Cliente con buen historial de pago."
        )

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.divider()
st.caption(
    "Modelo: Red Neuronal (64→32→16→3) · Preprocesamiento: LabelEncoder + SelectKBest + MinMaxScaler + PCA"
)
