from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "knn_house_price_model.pkl"
SCALER_PATH = APP_DIR / "scaler.pkl"

STATES = [
    "Johor",
    "Kedah",
    "Kuala Lumpur",
    "Melaka",
    "Pahang",
    "Penang",
    "Perak",
    "Sabah",
    "Sarawak",
    "Selangor",
]

PROPERTY_TYPES = [
    "Apartment",
    "Bungalow",
    "Condominium",
    "Flat",
    "Semi D",
    "Service Residence",
    "Terrace House",
    "Town House",
]


@st.cache_resource
def load_assets():
    """Load the trained model and numerical-feature scaler."""
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def build_input(model, scaler, state, property_type, tenure, median_psf, transactions):
    """Build one prediction row and align it to the model's training columns."""
    scaled = scaler.transform(np.array([[median_psf, transactions]], dtype=float))[0]
    values = {
        "Median_PSF": scaled[0],
        "Transactions": scaled[1],
        "Tenure_Encoded": 1 if tenure == "Freehold" else 0,
        f"State_{state}": 1,
        f"Type_Clean_{property_type}": 1,
    }

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        # Fallback for models trained without DataFrame column names.
        fallback = [
            "Median_PSF",
            "Transactions",
            "Tenure_Encoded",
            "State_Kuala Lumpur",
            "State_Selangor",
            "State_Penang",
            "State_Johor",
            "Type_Clean_Condominium",
            "Type_Clean_Apartment",
            "Type_Clean_Semi D",
        ]
        feature_names = fallback

    return pd.DataFrame(
        [{column: values.get(column, 0) for column in feature_names}],
        columns=feature_names,
    )


st.set_page_config(page_title="Malaysia House Price Predictor", page_icon="🏡")
st.title("🏡 Malaysia House Price Predictor")
st.write("Estimate a property's market price using the trained KNN model.")

missing = [path.name for path in (MODEL_PATH, SCALER_PATH) if not path.exists()]
if missing:
    st.error(
        "Missing deployment file(s): "
        + ", ".join(missing)
        + ". Add them to the same GitHub folder as app.py."
    )
    st.stop()

try:
    model, scaler = load_assets()
except Exception as exc:
    st.error(f"The model assets could not be loaded: {exc}")
    st.stop()

with st.form("prediction_form"):
    state = st.selectbox("State", STATES)
    property_type = st.selectbox("Property type", PROPERTY_TYPES)
    tenure = st.radio("Tenure", ["Freehold", "Leasehold"], horizontal=True)

    left, right = st.columns(2)
    with left:
        median_psf = st.number_input(
            "Price per square foot (RM)", min_value=50.0, max_value=3000.0, value=450.0
        )
    with right:
        transactions = st.number_input(
            "Recent area transactions", min_value=1, max_value=500, value=25
        )

    submitted = st.form_submit_button("Predict house price", use_container_width=True)

if submitted:
    try:
        input_frame = build_input(
            model,
            scaler,
            state,
            property_type,
            tenure,
            median_psf,
            transactions,
        )
        prediction = float(model.predict(input_frame)[0])
        st.success(f"Estimated market price: RM {prediction:,.2f}")
        st.caption("The result is an estimate produced by the supplied trained model.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

