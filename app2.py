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
    scaler   = joblib.load("minmax_scaler.joblib")
    pca      = joblib.load("modelo_pca.joblib")
    modelo   = load_model("modelo_red_neuronal.keras")
    return encoders, scaler, pca, modelo

encoders, scaler, pca, modelo = cargar_artefactos()

# ── Constantes ─────────────────────────────────────────────────────────────────
# Orden exacto de las 19 columnas con las que se entrenó el scaler
FEATURES_ORDER = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Credit_Mix", "Outstanding_Debt", "Credit_Utilization_Ratio",
    "Credit_History_Age", "Payment_of_Min_Amount", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance"
]

# Índices de las 5 features seleccionadas por SelectKBest dentro de FEATURES_ORDER
# Num_Credit_Card=4, Interest_Rate=5, Delay_from_due_date=7,
# Num_Credit_Inquiries=10, Payment_of_Min_Amount=15
SELECTED_INDICES = [4, 5, 7, 10, 15]

CLASES  = {0: "⚠️ Poor", 1: "🟡 Standard", 2: "✅ Good"}
COLORES = {0: "#FF4B4B", 1: "#FFA500",    2: "#21C354"}

# ── Interfaz ───────────────────────────────────────────────────────────────────
st.title("💳 Predictor de Riesgo de Crédito")
st.markdown("Ingresa los datos del cliente para predecir su **Credit Score**.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Información financiera")
    age                  = st.number_input("Edad",                          min_value=14.0,  max_value=56.0,   value=30.0,  step=0.5)
    annual_income        = st.number_input("Ingreso anual ($)",             min_value=7000.0, max_value=180000.0, value=50000.0, step=500.0)
    monthly_salary       = st.number_input("Salario mensual en mano ($)",   min_value=300.0,  max_value=15300.0,  value=4000.0,  step=100.0)
    num_bank_accounts    = st.number_input("Número de cuentas bancarias",   min_value=0.0,   max_value=11.0,   value=5.0,   step=0.5)
    num_credit_card      = st.number_input("Número de tarjetas de crédito", min_value=0.0,   max_value=11.0,   value=5.0,   step=0.5)
    interest_rate        = st.number_input("Tasa de interés (%)",           min_value=1,     max_value=34,     value=14,    step=1)
    num_of_loan          = st.number_input("Número de préstamos",           min_value=0,     max_value=9,      value=3,     step=1)
    delay_from_due_date  = st.number_input("Días de retraso en pagos",      min_value=-2.0,  max_value=64.0,   value=20.0,  step=0.5)
    num_delayed_payment  = st.number_input("Número de pagos retrasados",    min_value=0.0,   max_value=27.0,   value=13.0,  step=0.5)
    changed_credit_limit = st.number_input("Cambio en límite de crédito",   min_value=0.5,   max_value=32.0,   value=10.0,  step=0.5)

with col2:
    st.subheader("📋 Más datos")
    num_credit_inquiries  = st.number_input("Consultas de crédito",         min_value=0.0,   max_value=17.0,   value=5.0,   step=0.5)
    outstanding_debt      = st.number_input("Deuda pendiente ($)",          min_value=0.0,   max_value=5000.0, value=1400.0, step=10.0)
    credit_util_ratio     = st.number_input("Ratio utilización crédito (%)",min_value=25.0,  max_value=43.0,   value=32.0,  step=0.1)
    credit_history_age    = st.number_input("Antigüedad historial (años)",  min_value=0.0,   max_value=34.0,   value=18.0,  step=0.5)
    total_emi             = st.number_input("EMI total mensual ($)",        min_value=0.0,   max_value=1520.0, value=100.0, step=5.0)
    amount_invested       = st.number_input("Inversión mensual ($)",        min_value=14.0,  max_value=1010.0, value=190.0, step=5.0)
    monthly_balance       = st.number_input("Balance mensual ($)",          min_value=92.0,  max_value=1350.0, value=400.0, step=5.0)

    st.markdown("---")
    credit_mix            = st.selectbox("Mezcla de crédito",       options=["Bad", "Standard", "Good"], index=1)
    payment_of_min_amount = st.selectbox("¿Paga el monto mínimo?",  options=["Yes", "No", "NM"],         index=0)

st.divider()

# ── Predicción ─────────────────────────────────────────────────────────────────
if st.button("🔍 Predecir riesgo de crédito", use_container_width=True, type="primary"):

    # 1. Codificar columnas categóricas con LabelEncoder
    le_credit_mix = encoders["Credit_Mix"]
    le_payment    = encoders["Payment_of_Min_Amount"]

    try:
        credit_mix_enc = le_credit_mix.transform([credit_mix])[0]
    except ValueError:
        credit_mix_enc = 1  # Standard por defecto

    try:
        payment_enc = le_payment.transform([payment_of_min_amount])[0]
    except ValueError:
        payment_enc = le_payment.transform(["Yes"])[0]

    # 2. Construir vector con las 19 features en el orden correcto
    X_input = np.array([[
        age, annual_income, monthly_salary, num_bank_accounts,
        num_credit_card, interest_rate, num_of_loan, delay_from_due_date,
        num_delayed_payment, changed_credit_limit, num_credit_inquiries,
        credit_mix_enc, outstanding_debt, credit_util_ratio,
        credit_history_age, payment_enc, total_emi,
        amount_invested, monthly_balance
    ]])

    # 3. Seleccionar las 5 features (SelectKBest)
    X_selected = X_input[:, SELECTED_INDICES]

    # 4. Reducir con PCA a 3 componentes
    X_pca = pca.transform(X_selected)

    # 5. Escalar con MinMaxScaler (espera 3 features, después del PCA)
    X_scaled = scaler.transform(X_pca)

    # 6. Predecir
    probas    = modelo.predict(X_scaled, verbose=0)[0]
    clase     = int(np.argmax(probas))
    confianza = float(probas[clase]) * 100

    # ── Resultado ──
    st.subheader("📋 Resultado de la predicción")
    color = COLORES[clase]
    label = CLASES[clase]

    st.markdown(
        f"""
        <div style="background-color:{color}22; border-left:5px solid {color};
                    padding:16px; border-radius:8px; margin-bottom:16px;">
            <h2 style="color:{color}; margin:0;">{label}</h2>
            <p style="margin:4px 0 0 0; color:#555;">Confianza: <strong>{confianza:.1f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Distribución de probabilidades:**")
    prob_df = pd.DataFrame({
        "Clase":        [CLASES[i] for i in range(3)],
        "Probabilidad": [f"{p*100:.1f}%" for p in probas],
    })
    st.dataframe(prob_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    if clase == 0:
        st.error("**Poor** → Alto riesgo crediticio. Se recomienda mayor análisis antes de aprobar crédito.")
    elif clase == 1:
        st.warning("**Standard** → Riesgo moderado. El cliente puede ser elegible con condiciones estándar.")
    else:
        st.success("**Good** → Bajo riesgo crediticio. Cliente con buen historial de pago.")

st.divider()
st.caption("Modelo: Red Neuronal (64→32→16→3) · Pipeline: LabelEncoder → MinMaxScaler → SelectKBest(k=5) → PCA(3) → NN")
