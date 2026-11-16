"""Step 2 - Data preparation: clean, engineer features, split and save train/test sets."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/SuperKart.csv"
OUT_DIR = "artifacts"
TARGET = "Product_Store_Sales_Total"
REFERENCE_YEAR = 2023
PERISHABLE = {"Dairy", "Meat", "Fruits and Vegetables", "Breakfast",
              "Breads", "Seafood", "Frozen Foods"}


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {df.shape}")

    # --- Cleaning ---
    df = df.drop_duplicates()
    df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})

    # --- Feature engineering ---
    df["Product_Id_char"] = df["Product_Id"].str[:2]                       # FD / DR / NC
    df["Product_Type_Category"] = np.where(
        df["Product_Type"].isin(PERISHABLE), "Perishables", "Non Perishables")
    df["Store_Age_Years"] = REFERENCE_YEAR - df["Store_Establishment_Year"]

    # --- Drop unnecessary columns (identifiers / replaced by engineered features) ---
    df = df.drop(columns=["Product_Id", "Product_Type", "Store_Id", "Store_Establishment_Year"])
    print("Final columns:", list(df.columns))

    # --- Split ---
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    os.makedirs(OUT_DIR, exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv(f"{OUT_DIR}/train.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(f"{OUT_DIR}/test.csv", index=False)
    print(f"Saved train {X_train.shape} and test {X_test.shape} to {OUT_DIR}/")


if __name__ == "__main__":
    main()
