"""Streamlit app for SuperKart sales forecasting."""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒")


@st.cache_resource
def load_model():
    return joblib.load("models/best_model.joblib")


model = load_model()

st.title("🛒 SuperKart Sales Forecast")
st.write("Predict total sales of a product in a store.")

PERISHABLE = {"Dairy", "Meat", "Fruits and Vegetables", "Breakfast", "Breads", "Seafood", "Frozen Foods"}
REFERENCE_YEAR = 2023

col1, col2 = st.columns(2)
with col1:
    st.subheader("Product")
    weight = st.number_input("Product weight", 4.0, 22.0, 12.5, step=0.1)
    sugar = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
    area = st.number_input("Allocated display area ratio", 0.004, 0.30, 0.07, step=0.001, format="%.3f")
    ptype = st.selectbox("Product type", [
        "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods",
        "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household",
        "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"])
    mrp = st.number_input("Product MRP", 30.0, 270.0, 147.0, step=0.5)
    code = st.selectbox("Product ID prefix", ["FD", "NC", "DR"],
                        help="FD = Food, NC = Non-consumable, DR = Drinks")
with col2:
    st.subheader("Store")
    year = st.number_input("Store establishment year", 1987, 2009, 2002, step=1)
    size = st.selectbox("Store size", ["Small", "Medium", "High"])
    city = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"])
    stype = st.selectbox("Store type", [
        "Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])

# Same engineered features as the training pipeline
input_df = pd.DataFrame([{
    "Product_Weight": weight,
    "Product_Sugar_Content": sugar,
    "Product_Allocated_Area": area,
    "Product_MRP": mrp,
    "Store_Size": size,
    "Store_Location_City_Type": city,
    "Store_Type": stype,
    "Product_Id_char": code,
    "Product_Type_Category": "Perishables" if ptype in PERISHABLE else "Non Perishables",
    "Store_Age_Years": REFERENCE_YEAR - year,
}])

st.subheader("Input summary")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict sales", type="primary"):
    pred = model.predict(input_df)[0]
    st.success(f"Predicted total sales: {pred:,.2f}")
