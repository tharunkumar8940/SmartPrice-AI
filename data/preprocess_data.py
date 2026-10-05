from pathlib import Path

import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_FILE = BASE_DIR / "raw" / "marketplace_products.csv"
PROCESSED_DIR = BASE_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "marketplace_products_clean.csv"


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "product_id",
    "marketplace",
    "product_name",
    "brand",
    "category",
    "subcategory",
    "model_number",
    "variant",
    "color",
    "storage",
    "ram",
    "rating",
    "review_count",
    "original_price",
    "discount_percent",
    "selling_price",
    "shipping_cost",
    "coupon_discount",
    "effective_price",
    "availability",
    "product_url",
    "scraped_at",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():
    """Load the raw marketplace dataset."""

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_FILE}\n\n"
            "Run create_dataset.py first."
        )

    df = pd.read_csv(RAW_FILE)

    if df.empty:
        raise ValueError("The raw dataset is empty.")

    return df


# ============================================================
# VALIDATE COLUMNS
# ============================================================

def validate_columns(df):
    """Check that all required columns exist."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text_columns(df):
    """Clean text fields consistently."""

    text_columns = [
        "product_id",
        "marketplace",
        "product_name",
        "brand",
        "category",
        "subcategory",
        "model_number",
        "variant",
        "color",
        "storage",
        "ram",
        "availability",
        "product_url",
    ]

    for column in text_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    return df


# ============================================================
# CLEAN NUMERIC DATA
# ============================================================

def clean_numeric_columns(df):
    """Convert numerical fields to numeric data types."""

    numeric_columns = [
        "rating",
        "review_count",
        "original_price",
        "discount_percent",
        "selling_price",
        "shipping_cost",
        "coupon_discount",
        "effective_price",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


# ============================================================
# CLEAN DATE
# ============================================================

def clean_date_column(df):
    """Convert scraped_at into a proper datetime column."""

    df["scraped_at"] = pd.to_datetime(
        df["scraped_at"],
        errors="coerce",
    )

    if df["scraped_at"].isna().any():
        raise ValueError(
            "Invalid dates found in scraped_at."
        )

    return df


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df):
    """Handle missing values without corrupting price information."""

    # Price fields may legitimately be missing
    # when a product is out of stock.
    price_columns = [
        "selling_price",
        "effective_price",
    ]

    # Text fields should not remain missing.
    text_fill_columns = [
        "variant",
        "color",
        "storage",
        "ram",
    ]

    for column in text_fill_columns:
        df[column] = df[column].fillna("Unknown")

    # Ratings and review counts can safely use neutral defaults
    # if missing.
    df["rating"] = df["rating"].fillna(
        df["rating"].median()
    )

    df["review_count"] = df["review_count"].fillna(0)

    # Keep price fields as NaN when unavailable.
    for column in price_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):
    """Remove duplicate marketplace observations."""

    duplicate_subset = [
        "product_id",
        "marketplace",
        "scraped_at",
    ]

    before = len(df)

    df = df.drop_duplicates(
        subset=duplicate_subset,
        keep="first",
    ).reset_index(drop=True)

    removed = before - len(df)

    print(f"Duplicate records removed: {removed}")

    return df


# ============================================================
# DATA QUALITY RULES
# ============================================================

def validate_data_quality(df):
    """Apply business-level data quality checks."""

    valid_marketplaces = {
        "Amazon",
        "Flipkart",
        "Meesho",
    }

    detected_marketplaces = set(
        df["marketplace"].dropna().unique()
    )

    unexpected = (
        detected_marketplaces
        - valid_marketplaces
    )

    if unexpected:
        raise ValueError(
            f"Unexpected marketplaces found: {unexpected}"
        )

    # Ratings must be between 0 and 5.
    invalid_ratings = (
        df["rating"].notna()
        & (
            (df["rating"] < 0)
            | (df["rating"] > 5)
        )
    )

    if invalid_ratings.any():
        raise ValueError(
            "Invalid rating values detected."
        )

    # Discounts should be between 0 and 100.
    invalid_discounts = (
        df["discount_percent"].notna()
        & (
            (df["discount_percent"] < 0)
            | (df["discount_percent"] > 100)
        )
    )

    if invalid_discounts.any():
        raise ValueError(
            "Invalid discount percentages detected."
        )

    # Monetary values cannot be negative.
    monetary_columns = [
        "original_price",
        "selling_price",
        "shipping_cost",
        "coupon_discount",
        "effective_price",
    ]

    for column in monetary_columns:
        invalid_values = (
            df[column].notna()
            & (df[column] < 0)
        )

        if invalid_values.any():
            raise ValueError(
                f"Negative values detected in {column}."
            )

    return True


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):
    """Create reusable analytical features."""

    # Price difference between original and selling price.
    df["discount_amount"] = (
        df["original_price"]
        - df["selling_price"]
    )

    # Total customer-facing cost after shipping and coupon.
    df["customer_payable_price"] = (
        df["selling_price"]
        + df["shipping_cost"]
        - df["coupon_discount"]
    )

    # Difference between selling price and effective price.
    df["price_adjustment"] = (
        df["effective_price"]
        - df["selling_price"]
    )

    # Date-based features for price history analysis.
    df["year"] = df["scraped_at"].dt.year
    df["month"] = df["scraped_at"].dt.month
    df["day"] = df["scraped_at"].dt.day
    df["day_of_week"] = (
        df["scraped_at"].dt.dayofweek
    )

    # Weekend indicator.
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    return df


# ============================================================
# FINAL SORTING
# ============================================================

def sort_dataset(df):
    """Sort records for consistent downstream processing."""

    return df.sort_values(
        [
            "product_id",
            "marketplace",
            "scraped_at",
        ]
    ).reset_index(drop=True)


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(df):
    """Save the cleaned dataset."""

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return OUTPUT_FILE


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 65)
    print("SMARTPRICE AI - DATA PREPROCESSING PIPELINE")
    print("=" * 65)

    # Load
    df = load_dataset()

    print(f"\nRaw records: {len(df):,}")
    print(f"Raw columns: {len(df.columns)}")

    # Validate structure
    validate_columns(df)

    # Clean
    df = clean_text_columns(df)
    df = clean_numeric_columns(df)
    df = clean_date_column(df)

    # Missing values
    df = handle_missing_values(df)

    # Duplicate removal
    df = remove_duplicates(df)

    # Business validation
    validate_data_quality(df)

    # Feature engineering
    df = create_features(df)

    # Sorting
    df = sort_dataset(df)

    # Save
    output_path = save_dataset(df)

    print("\nProcessing completed successfully.")

    print(f"\nClean records: {len(df):,}")
    print(f"Final columns: {len(df.columns)}")

    print("\nMarketplace distribution:")
    print(
        df["marketplace"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nDate range:")
    print(
        f"{df['scraped_at'].min().date()} "
        f"to "
        f"{df['scraped_at'].max().date()}"
    )

    print("\nProcessed dataset:")
    print(output_path)

    print("\n" + "=" * 65)
    print("PREPROCESSING PIPELINE COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()