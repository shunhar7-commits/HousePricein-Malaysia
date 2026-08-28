import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Malaysia House Price Prediction",
    page_icon="🏡",
    layout="wide",
)

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
        candidate = APP_DIR / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Upload malaysia_house_price_data_2025.csv to the same GitHub folder as app.py."
    )


@st.cache_data
def load_data():
    original = pd.read_csv(find_dataset())
    required = ["State", "Tenure", "Type", TARGET, *NUMERIC_FEATURES]
    missing = [column for column in required if column not in original.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    clean = original.copy()
    clean["Type_Clean"] = clean["Type"].astype(str).str.split(",").str[0].str.strip()
    for column in [TARGET, *NUMERIC_FEATURES]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=FEATURES + [TARGET]).drop_duplicates()
    return original, clean


def create_pipeline():
    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "regressor",
                KNeighborsRegressor(
                    n_neighbors=7, weights="distance", metric="euclidean"
                ),
            ),
        ]
    )


@st.cache_resource(show_spinner="Training the KNN model...")
def train_model(clean):
    X_train, X_test, y_train, y_test = train_test_split(
        clean[FEATURES], clean[TARGET], test_size=0.20, random_state=42
    )
    evaluation_model = create_pipeline()
    evaluation_model.fit(X_train, y_train)
    prediction = evaluation_model.predict(X_test)
    metrics = {
        "R²": r2_score(y_test, prediction),
        "RMSE": np.sqrt(mean_squared_error(y_test, prediction)),
        "MAE": mean_absolute_error(y_test, prediction),
    }

    final_model = create_pipeline()
    final_model.fit(clean[FEATURES], clean[TARGET])
    return final_model, y_test, prediction, metrics


def prediction_frame(state, property_type, tenure, median_psf, transactions):
    return pd.DataFrame(
        [{
            "Median_PSF": median_psf,
            "Transactions": transactions,
            "Tenure": tenure,
            "State": state,
            "Type_Clean": property_type,
        }]
    )


try:
    df_original, df_clean = load_data()
    model, y_test, test_prediction, model_metrics = train_model(df_clean)
except Exception as error:
    st.error("The application could not prepare the dataset or model.")
    st.exception(error)
    st.stop()


states = sorted(df_clean["State"].astype(str).unique())
property_types = sorted(df_clean["Type_Clean"].astype(str).unique())
tenures = sorted(df_clean["Tenure"].astype(str).unique())

st.sidebar.title("🏡 House Price Prediction")
st.sidebar.caption("Malaysia property analytics • 2025")
st.sidebar.divider()
st.sidebar.subheader("Property Information")

state = st.sidebar.selectbox("State", states)
property_type = st.sidebar.selectbox("Property Type", property_types)
tenure = st.sidebar.selectbox("Tenure", tenures)
median_psf = st.sidebar.slider(
    "Median Price per Sq. Ft. (RM)",
    min_value=int(df_clean["Median_PSF"].min()),
    max_value=int(df_clean["Median_PSF"].max()),
    value=int(df_clean["Median_PSF"].median()),
)
transactions = st.sidebar.slider(
    "Recent Transactions",
    min_value=max(1, int(df_clean["Transactions"].min())),
    max_value=int(df_clean["Transactions"].max()),
    value=max(1, int(df_clean["Transactions"].median())),
)
st.sidebar.divider()
predict_clicked = st.sidebar.button(
    "Predict House Price", type="primary", use_container_width=True
)

st.title("Malaysia House Price Prediction")
st.caption(
    "AI-assisted property price estimation with KNN modelling, exploratory analysis, "
    "performance evaluation, and batch prediction."
)

prediction_tab, exploration_tab, performance_tab, batch_tab, about_tab = st.tabs(
    ["Prediction", "Data Exploration", "Model Performance", "Batch Prediction", "About"]
)


with prediction_tab:
    st.header("Property Price Assessment")
    st.markdown("#### How to Use")
    step1, step2, step3 = st.columns(3)
    step1.info("**Step 1 — Enter Property Data**\n\nUse the sidebar to select the location and property characteristics.")
    step2.info("**Step 2 — Click Predict**\n\nPress the prediction button at the bottom of the sidebar.")
    step3.info("**Step 3 — Review Results**\n\nView the estimate, property profile, and comparison with market medians.")

    if predict_clicked:
        input_data = prediction_frame(
            state, property_type, tenure, median_psf, transactions
        )
        predicted_price = max(0.0, float(model.predict(input_data)[0]))
        overall_median = float(df_clean[TARGET].median())
        difference = predicted_price - overall_median

        st.divider()
        result1, result2, result3 = st.columns(3)
        result1.metric("Estimated Median Price", f"RM {predicted_price:,.0f}")
        result2.metric(
            "Difference from Dataset Median",
            f"RM {difference:,.0f}",
            delta=f"{difference / overall_median * 100:+.1f}%",
        )
        result3.metric("Price per Sq. Ft.", f"RM {median_psf:,.0f}")

        st.subheader("Property Profile")
        st.dataframe(
            pd.DataFrame([{
                "State": state,
                "Property Type": property_type,
                "Tenure": tenure,
                "Median PSF (RM)": median_psf,
                "Transactions": transactions,
            }]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("This result is a statistical estimate, not a professional valuation.")
    else:
        st.info("Awaiting property data — complete the sidebar and click Predict House Price.")


with exploration_tab:
    st.header("Data Exploration")
    st.caption("Exploratory analysis of the Malaysia House Price Dataset 2025")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Records", f"{len(df_clean):,}")
    m2.metric("Total Columns", f"{df_clean.shape[1]:,}")
    m3.metric("States", f"{df_clean['State'].nunique():,}")
    m4.metric("Property Types", f"{df_clean['Type_Clean'].nunique():,}")
    m5.metric("Median Price", f"RM {df_clean[TARGET].median():,.0f}")

    removed = len(df_original) - len(df_clean)
    st.info(
        f"**Data cleaning note:** {removed:,} incomplete or duplicate record(s) were removed. "
        f"The cleaned dataset contains {len(df_clean):,} records."
    )

    with st.expander("Column Details"):
        st.markdown("""
        - **Township / Area / State:** Geographic property location
        - **Tenure:** Freehold or leasehold ownership
        - **Type:** Property category
        - **Median_Price:** Target median property price in RM
        - **Median_PSF:** Median price per square foot
        - **Transactions:** Number of recorded market transactions
        """)

    chart1, chart2 = st.columns(2)
    with chart1:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df_clean[TARGET], bins=35, kde=True, color="teal", ax=ax)
        ax.set_xlabel("Median Price (RM)")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with chart2:
        st.subheader("Feature Correlation")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            df_clean[[TARGET, "Median_PSF", "Transactions"]].corr(),
            annot=True, cmap="YlGnBu", fmt=".2f", ax=ax
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("Feature Distributions")
    selected_numeric = st.selectbox(
        "Select numeric feature", [TARGET, "Median_PSF", "Transactions"],
        key="numeric_feature"
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=df_clean, x=selected_numeric, color="skyblue", ax=ax)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    selected_category = st.selectbox(
        "Select categorical feature", ["State", "Tenure", "Type_Clean"],
        key="category_feature"
    )
    order = df_clean[selected_category].value_counts().index
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df_clean, y=selected_category, order=order, color="teal", ax=ax)
    ax.set_xlabel("Number of Records")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("Data Explorer — Full Dataset"):
        state_filter = st.multiselect("Filter by state", states)
        displayed = df_clean[
            df_clean["State"].isin(state_filter)
        ] if state_filter else df_clean
        st.dataframe(displayed, use_container_width=True)
    with st.expander("Statistical Summary"):
        st.dataframe(df_clean.describe(include="all").transpose(), use_container_width=True)


with performance_tab:
    st.header("Model Performance")
    st.caption("KNN test-set evaluation using a fixed 80/20 split (random state 42)")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("R² Score", f"{model_metrics['R²']:.4f}")
    p2.metric("RMSE", f"RM {model_metrics['RMSE']:,.2f}")
    p3.metric("MAE", f"RM {model_metrics['MAE']:,.2f}")
    p4.metric("Test Records", f"{len(y_test):,}")

    if model_metrics["R²"] >= 0.6:
        st.success("The model explains a substantial share of test-set price variation.")
    else:
        st.warning("Model accuracy is moderate; predictions should be interpreted cautiously.")

    left, right = st.columns(2)
    with left:
        st.subheader("Actual vs Predicted Prices")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_test, test_prediction, alpha=0.55, color="teal")
        low = min(y_test.min(), test_prediction.min())
        high = max(y_test.max(), test_prediction.max())
        ax.plot([low, high], [low, high], "r--", label="Perfect prediction")
        ax.set_xlabel("Actual Price (RM)")
        ax.set_ylabel("Predicted Price (RM)")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.subheader("Residual Distribution")
        residuals = y_test.to_numpy() - test_prediction
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(residuals, bins=30, kde=True, color="coral", ax=ax)
        ax.axvline(0, color="black", linestyle="--")
        ax.set_xlabel("Actual − Predicted Price (RM)")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with st.expander("Model Configuration"):
        st.code(
            "Algorithm: KNeighborsRegressor\n"
            "Neighbors: 7\nWeights: distance\nMetric: euclidean\n"
            "Numerical preprocessing: StandardScaler\n"
            "Categorical preprocessing: OneHotEncoder(handle_unknown='ignore')"
        )


with batch_tab:
    st.header("Batch Prediction")
    st.write("Upload a CSV containing the five required prediction columns.")
    st.code("Median_PSF, Transactions, Tenure, State, Type_Clean")

    template = pd.DataFrame([{column: "" for column in FEATURES}])
    st.download_button(
        "Download CSV Template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="house_price_prediction_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload prediction CSV", type="csv")
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            missing = [column for column in FEATURES if column not in batch.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            batch["Predicted_Median_Price"] = model.predict(batch[FEATURES])
            st.success(f"Generated {len(batch):,} predictions.")
            st.dataframe(batch, use_container_width=True)
            st.download_button(
                "Download Predictions",
                data=batch.to_csv(index=False).encode("utf-8"),
                file_name="house_price_predictions.csv",
                mime="text/csv",
            )
        except Exception as error:
            st.error(f"Batch prediction failed: {error}")


with about_tab:
    st.header("About This Project")
    st.markdown("""
    ### Purpose
    This application demonstrates an end-to-end machine-learning workflow for analysing
    and predicting Malaysian residential property prices.

    ### Methodology
    1. Load and validate the 2025 Malaysia house-price dataset.
    2. Clean property types and remove incomplete or duplicate records.
    3. Scale numerical inputs and one-hot encode categorical inputs.
    4. Train a seven-neighbour KNN regression model.
    5. Evaluate the model on an unseen 20% test set.
    6. Retrain on the full dataset for individual and batch predictions.

    ### Important Limitations
    - The output is an estimate based only on patterns in the supplied dataset.
    - The model does not include property size, age, condition, exact address, or amenities.
    - Market conditions can change after the dataset collection period.
    - Results should not replace a valuation by a licensed property professional.

    ### Technology
    Python, Streamlit, pandas, scikit-learn, Matplotlib, and Seaborn.
    """)

