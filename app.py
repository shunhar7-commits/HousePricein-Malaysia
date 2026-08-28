from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APP_DIR = Path(__file__).resolve().parent
DATASET_NAMES = (
    "malaysia_house_price_data_2025.csv",
    "malaysia_house_price_data_2025 (1).csv",
)
FEATURES = ["Median_PSF", "Transactions", "Tenure", "State", "Type_Clean"]
NUMERIC_FEATURES = ["Median_PSF", "Transactions"]
CATEGORICAL_FEATURES = ["Tenure", "State", "Type_Clean"]
TARGET = "Median_Price"


def find_dataset():
    for name in DATASET_NAMES:
        path = APP_DIR / name
        if path.exists():
            return path
    return None


@st.cache_resource(show_spinner="Training the KNN model...")
def train_model(dataset_path):
    data = pd.read_csv(dataset_path)
    required = {"Type", TARGET, *FEATURES[:-1]}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(missing))

    data = data.copy()
    data["Type_Clean"] = data["Type"].astype(str).str.split(",").str[0].str.strip()
    data = data.dropna(subset=FEATURES + [TARGET])

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("regressor", KNeighborsRegressor(n_neighbors=7, weights="distance")),
        ]
    )
    model.fit(data[FEATURES], data[TARGET])
    return model, data


st.set_page_config(page_title="Malaysia House Price Predictor", page_icon="🏡")
st.title("🏡 Malaysia House Price Predictor")
st.write("Estimate a property's median market price using a KNN regression model.")

dataset_path = find_dataset()
if dataset_path is None:
    st.error(
        "Dataset not found. Upload `malaysia_house_price_data_2025.csv` "
        "to the same GitHub folder as `app.py`."
    )
    st.stop()

try:
    model, dataset = train_model(dataset_path)
except Exception as error:
    st.error(f"The model could not be prepared: {error}")
    st.stop()

states = sorted(dataset["State"].dropna().astype(str).unique())
tenures = sorted(dataset["Tenure"].dropna().astype(str).unique())
property_types = sorted(dataset["Type_Clean"].dropna().astype(str).unique())

with st.form("prediction_form"):
    state = st.selectbox("State", states)
    property_type = st.selectbox("Property type", property_types)
    tenure = st.radio("Tenure", tenures, horizontal=True)

    left, right = st.columns(2)
    with left:
        median_psf = st.number_input(
            "Median price per square foot (RM)",
            min_value=0.0,
            value=float(dataset["Median_PSF"].median()),
            step=10.0,
        )
    with right:
        transactions = st.number_input(
            "Recent transactions",
            min_value=1,
            value=max(1, int(dataset["Transactions"].median())),
            step=1,
        )

    submitted = st.form_submit_button(
        "Predict house price", type="primary", use_container_width=True
    )

if submitted:
    input_data = pd.DataFrame(
        [{
            "Median_PSF": median_psf,
            "Transactions": transactions,
            "Tenure": tenure,
            "State": state,
            "Type_Clean": property_type,
        }]
    )
    try:
        prediction = float(model.predict(input_data)[0])
        st.success(f"Estimated median market price: RM {prediction:,.2f}")
        st.caption(
            "This estimate uses the supplied 2025 dataset and is not a valuation guarantee."
        )
    except Exception as error:
        st.error(f"Prediction failed: {error}")

with st.expander("Model information"):
    st.write(f"Training records: {len(dataset):,}")
    st.write("Algorithm: K-Nearest Neighbors regression (7 neighbors)")
    st.write(f"Dataset: {dataset_path.name}")

