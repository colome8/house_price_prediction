"""Streamlit app: predict the sale price of a house in Ames, Iowa.

Run (from the project root):
    python -m src.train        # once, to create models/
    streamlit run app.py
"""

import streamlit as st

from src.predict import MODEL_DIR, build_house, load_artifacts, predict_price

NEIGHBORHOODS = {
    "Bloomington Heights": "Blmngtn", "Bluestem": "Blueste", "Briardale": "BrDale",
    "Brookside": "BrkSide", "Clear Creek": "ClearCr", "College Creek": "CollgCr",
    "Crawford": "Crawfor", "Edwards": "Edwards", "Gilbert": "Gilbert",
    "Iowa DOT and Rail Road": "IDOTRR", "Meadow Village": "MeadowV", "Mitchell": "Mitchel",
    "North Ames": "NAmes", "Northpark Villa": "NPkVill", "Northridge": "NoRidge",
    "Northridge Heights": "NridgHt", "Northwest Ames": "NWAmes", "Old Town": "OldTown",
    "Sawyer": "Sawyer", "Sawyer West": "SawyerW", "Somerset": "Somerst",
    "South & West of Iowa State University": "SWISU", "Stone Brook": "StoneBr",
    "Timberland": "Timber", "Veenker": "Veenker",
}
KITCHEN = {"Excellent": "Ex", "Good": "Gd", "Typical / Average": "TA", "Fair": "Fa"}

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")
st.caption("Estimate the sale price of a house in Ames, Iowa (2006–2010 market). "
           "Model: average of Lasso + XGBoost, typical error ~7.5%.")


@st.cache_resource
def get_model():
    return load_artifacts()


if not (MODEL_DIR / "final_model.joblib").exists():
    st.error("Model not found. Train it first with: `python -m src.train`")
    st.stop()
model, typical = get_model()

with st.form("house"):
    neighborhood = st.selectbox("Neighborhood", list(NEIGHBORHOODS), index=list(NEIGHBORHOODS).index("North Ames"))

    c1, c2 = st.columns(2)
    overall_qual = c1.slider("Overall quality (1 = very poor, 10 = excellent)", 1, 10, 6)
    overall_cond = c2.slider("Overall condition (1 = very poor, 10 = excellent)", 1, 10, 5)

    c1, c2, c3 = st.columns(3)
    living_area = c1.number_input("Living area (sq ft)", 300, 5000, 1500, step=50)
    basement_area = c2.number_input("Basement area (sq ft)", 0, 3000, 1000, step=50, help="0 = no basement")
    lot_area = c3.number_input("Lot area (sq ft)", 1000, 100_000, 9500, step=500)

    c1, c2, c3 = st.columns(3)
    year_built = c1.number_input("Year built", 1872, 2010, 1975)
    floors = c2.radio("Floors", [1, 2], horizontal=True)
    kitchen = c3.selectbox("Kitchen quality", list(KITCHEN), index=2)

    c1, c2, c3 = st.columns(3)
    full_baths = c1.number_input("Full bathrooms", 0, 4, 2)
    garage_cars = c2.number_input("Garage (cars)", 0, 4, 2, help="0 = no garage")
    fireplaces = c3.number_input("Fireplaces", 0, 3, 1)

    submitted = st.form_submit_button("Predict price", type="primary", use_container_width=True)

if submitted:
    house = build_house(
        typical, neighborhood=NEIGHBORHOODS[neighborhood], overall_qual=overall_qual,
        overall_cond=overall_cond, living_area=living_area, floors=floors,
        basement_area=basement_area, lot_area=lot_area, year_built=year_built,
        full_baths=full_baths, garage_cars=garage_cars, kitchen_qual=KITCHEN[kitchen],
        fireplaces=fireplaces,
    )
    price = predict_price(model, house)

    st.metric("Estimated price", f"${price:,.0f}")
    st.write(f"Likely range: **\\${price * 0.9:,.0f} – \\${price * 1.1:,.0f}** "
             "(77% of houses in validation were predicted within ±10%).")
    st.caption("Features not shown in the form are filled with the values of a typical house in the dataset.")
