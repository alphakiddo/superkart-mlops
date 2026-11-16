"""Step 1 - Data registration: validate the raw dataset and print a summary."""
import sys
import pandas as pd

DATA_PATH = "data/SuperKart.csv"

EXPECTED_COLUMNS = [
    "Product_Id", "Product_Weight", "Product_Sugar_Content",
    "Product_Allocated_Area", "Product_Type", "Product_MRP", "Store_Id",
    "Store_Establishment_Year", "Store_Size", "Store_Location_City_Type",
    "Store_Type", "Product_Store_Sales_Total",
]


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing:
        print(f"VALIDATION FAILED - missing columns: {missing}")
        sys.exit(1)
    if extra:
        print(f"Warning - unexpected extra columns: {extra}")

    print("VALIDATION PASSED - all expected columns present")
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nData types:")
    print(df.dtypes)
    print(f"\nMissing values: {int(df.isna().sum().sum())}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")
    print("\nNumeric summary:")
    print(df.describe().T)


if __name__ == "__main__":
    main()
