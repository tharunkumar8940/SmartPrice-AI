"""
SmartPrice-AI
Price Normalization Engine

Purpose:
    Normalize marketplace prices into a comparable
    customer-payable price.

Key rules:
    1. Never treat missing selling prices as zero.
    2. Never use out-of-stock records for current comparison.
    3. Select the latest valid in-stock record for each
       product + marketplace combination.
    4. Calculate normalized price consistently.
    5. Preserve historical data separately.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"

INPUT_FILE = PROCESSED_DIR / "marketplace_products_clean.csv"

NORMALIZED_OUTPUT = (
    PROCESSED_DIR / "normalized_marketplace_prices.csv"
)

COMPARISON_OUTPUT = (
    PROCESSED_DIR / "marketplace_price_comparison.csv"
)


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
# DISPLAY
# ============================================================

def print_header():
    """Display module header."""

    print()
    print("=" * 70)
    print("SMARTPRICE-AI - PRICE NORMALIZATION ENGINE")
    print("=" * 70)
    print()


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load and validate the processed dataset."""

    print("Loading processed dataset...")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found:\n{INPUT_FILE}\n\n"
            "Run preprocess_data.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Input records: {len(df):,}")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"- {column}" for column in missing_columns)
        )

    print("Input validation: PASSED")
    print()

    return df


# ============================================================
# CLEAN NUMERIC COLUMNS
# ============================================================

def clean_numeric_columns(df):
    """Convert price-related columns to numeric values."""

    numeric_columns = [
        "original_price",
        "discount_percent",
        "selling_price",
        "shipping_cost",
        "coupon_discount",
        "effective_price",
        "rating",
        "review_count",
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

def clean_dates(df):
    """Convert scraped_at into datetime."""

    df["scraped_at"] = pd.to_datetime(
        df["scraped_at"],
        errors="coerce",
    )

    if df["scraped_at"].isna().any():
        invalid_count = int(df["scraped_at"].isna().sum())

        raise ValueError(
            f"Found {invalid_count} records with invalid scraped_at dates."
        )

    return df


# ============================================================
# PRICE VALIDATION
# ============================================================

def validate_price_data(df):
    """
    Validate price information.

    Important:
        Missing selling_price values are allowed in the historical
        dataset because marketplace products can become out of stock.

        However, those rows must NOT be used for current price
        comparison.
    """

    negative_selling = (
        df["selling_price"].notna()
        & (df["selling_price"] < 0)
    )

    if negative_selling.any():
        raise ValueError(
            "Negative selling prices detected."
        )

    negative_shipping = (
        df["shipping_cost"].notna()
        & (df["shipping_cost"] < 0)
    )

    if negative_shipping.any():
        raise ValueError(
            "Negative shipping costs detected."
        )

    negative_coupon = (
        df["coupon_discount"].notna()
        & (df["coupon_discount"] < 0)
    )

    if negative_coupon.any():
        raise ValueError(
            "Negative coupon discounts detected."
        )

    print("Price validation: PASSED")
    print()


# ============================================================
# SELECT LATEST VALID IN-STOCK RECORD
# ============================================================

def select_latest_valid_records(df):
    """
    Select the latest usable marketplace price.

    A record is considered valid for current comparison when:

        - product_id exists
        - marketplace exists
        - selling_price exists
        - selling_price > 0
        - availability is In Stock

    This prevents an out-of-stock record with NaN selling_price
    from becoming a fake ₹0/₹75 marketplace price.
    """

    print("Selecting latest valid marketplace observations...")

    valid = df[
        df["product_id"].notna()
        & df["marketplace"].notna()
        & df["selling_price"].notna()
        & (df["selling_price"] > 0)
        & (
            df["availability"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "in stock"
        )
    ].copy()

    if valid.empty:
        raise ValueError(
            "No valid in-stock marketplace price records were found."
        )

    valid = valid.sort_values(
        by=[
            "product_id",
            "marketplace",
            "scraped_at",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    )

    latest = (
        valid
        .groupby(
            ["product_id", "marketplace"],
            as_index=False,
            group_keys=False,
        )
        .tail(1)
        .copy()
    )

    latest = latest.sort_values(
        by=["product_id", "marketplace"]
    ).reset_index(drop=True)

    print(
        "Latest valid product-marketplace records: "
        f"{len(latest):,}"
    )

    print()

    return latest


# ============================================================
# CALCULATE NORMALIZED PRICE
# ============================================================

def calculate_normalized_prices(df):
    """
    Calculate the actual comparable customer-payable price.

    Formula:

        normalized_price =
            selling_price
            + shipping_cost
            - coupon_discount

    Coupon and shipping values are safely handled when missing.
    """

    print("Calculating normalized customer-payable prices...")

    df = df.copy()

    # Missing shipping means no additional shipping charge.
    df["shipping_cost"] = (
        df["shipping_cost"]
        .fillna(0)
        .clip(lower=0)
    )

    # Missing coupon means no coupon.
    df["coupon_discount"] = (
        df["coupon_discount"]
        .fillna(0)
        .clip(lower=0)
    )

    # Selling price must be valid at this stage.
    if df["selling_price"].isna().any():
        raise ValueError(
            "Missing selling prices found in current comparison records."
        )

    if (df["selling_price"] <= 0).any():
        raise ValueError(
            "Zero or negative selling prices found in current comparison records."
        )

    # A coupon cannot exceed the selling price.
    df["usable_coupon_discount"] = np.minimum(
        df["coupon_discount"],
        df["selling_price"],
    )

    # Customer payable price.
    df["normalized_price"] = (
        df["selling_price"]
        + df["shipping_cost"]
        - df["usable_coupon_discount"]
    )

    # Safety validation.
    if (df["normalized_price"] <= 0).any():
        raise ValueError(
            "Invalid normalized prices detected."
        )

    # Existing effective price from the dataset.
    df["dataset_effective_price"] = df[
        "effective_price"
    ]

    # Difference between our calculation and source effective price.
    df["normalization_difference"] = (
        df["normalized_price"]
        - df["dataset_effective_price"]
    )

    # Amount saved compared with original price.
    df["total_savings"] = (
        df["original_price"]
        - df["normalized_price"]
    )

    # Effective discount percentage.
    df["effective_discount_percent"] = np.where(
        df["original_price"] > 0,
        (
            df["total_savings"]
            / df["original_price"]
        )
        * 100,
        0,
    )

    # Prevent negative savings from being reported as a discount.
    df["effective_discount_percent"] = (
        df["effective_discount_percent"]
        .clip(lower=0)
    )

    return df


# ============================================================
# CREATE COMPARISON TABLE
# ============================================================

def create_comparison(df):
    """
    Create one comparison row per product with
    marketplace prices.
    """

    print("Creating marketplace comparison...")

    comparison_rows = []

    for product_id, group in df.groupby(
        "product_id",
        sort=True,
    ):
        product_name = group["product_name"].iloc[0]
        category = group["category"].iloc[0]
        brand = group["brand"].iloc[0]

        marketplace_prices = {}

        for _, row in group.iterrows():
            marketplace = row["marketplace"]

            marketplace_prices[marketplace] = {
                "price": float(row["normalized_price"]),
                "product_name": row["product_name"],
                "selling_price": float(
                    row["selling_price"]
                ),
                "shipping": float(
                    row["shipping_cost"]
                ),
                "coupon": float(
                    row["usable_coupon_discount"]
                ),
                "rating": row["rating"],
                "scraped_at": row["scraped_at"],
            }

        if not marketplace_prices:
            continue

        # Find lowest marketplace.
        lowest_marketplace = min(
            marketplace_prices,
            key=lambda marketplace:
                marketplace_prices[marketplace]["price"],
        )

        lowest_price = marketplace_prices[
            lowest_marketplace
        ]["price"]

        highest_price = max(
            item["price"]
            for item in marketplace_prices.values()
        )

        price_difference = (
            highest_price
            - lowest_price
        )

        potential_savings_percent = (
            (
                price_difference
                / highest_price
            )
            * 100
            if highest_price > 0
            else 0
        )

        row = {
            "product_id": product_id,
            "product_name": product_name,
            "brand": brand,
            "category": category,
            "lowest_price": round(
                lowest_price,
                2,
            ),
            "lowest_marketplace": (
                lowest_marketplace
            ),
            "highest_price": round(
                highest_price,
                2,
            ),
            "price_difference": round(
                price_difference,
                2,
            ),
            "potential_savings_percent": round(
                potential_savings_percent,
                2,
            ),
        }

        # Add marketplace-specific prices.
        for marketplace in [
            "Amazon",
            "Flipkart",
            "Meesho",
        ]:
            data = marketplace_prices.get(
                marketplace
            )

            if data:
                row[
                    f"{marketplace.lower()}_price"
                ] = round(
                    data["price"],
                    2,
                )

                row[
                    f"{marketplace.lower()}_selling_price"
                ] = round(
                    data["selling_price"],
                    2,
                )

                row[
                    f"{marketplace.lower()}_shipping"
                ] = round(
                    data["shipping"],
                    2,
                )

                row[
                    f"{marketplace.lower()}_coupon"
                ] = round(
                    data["coupon"],
                    2,
                )

                row[
                    f"{marketplace.lower()}_rating"
                ] = data["rating"]

            else:
                row[
                    f"{marketplace.lower()}_price"
                ] = np.nan

                row[
                    f"{marketplace.lower()}_selling_price"
                ] = np.nan

                row[
                    f"{marketplace.lower()}_shipping"
                ] = np.nan

                row[
                    f"{marketplace.lower()}_coupon"
                ] = np.nan

                row[
                    f"{marketplace.lower()}_rating"
                ] = np.nan

        comparison_rows.append(row)

    comparison = pd.DataFrame(
        comparison_rows
    )

    if comparison.empty:
        raise ValueError(
            "No marketplace comparison records were created."
        )

    comparison = comparison.sort_values(
        by="product_id"
    ).reset_index(drop=True)

    return comparison


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    normalized_df,
    comparison_df,
):
    """Save normalized and comparison datasets."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized_df.to_csv(
        NORMALIZED_OUTPUT,
        index=False,
    )

    comparison_df.to_csv(
        COMPARISON_OUTPUT,
        index=False,
    )

    print()
    print("PRICE NORMALIZATION COMPLETE")
    print()

    print("Normalized file:")
    print(NORMALIZED_OUTPUT)
    print()

    print("Comparison file:")
    print(COMPARISON_OUTPUT)
    print()

    print(
        f"Products compared: "
        f"{len(comparison_df):,}"
    )


# ============================================================
# DISPLAY SUMMARY
# ============================================================

def display_summary(
    normalized_df,
    comparison_df,
):
    """Display important price intelligence results."""

    print()
    print("-" * 70)
    print("PRICE INTELLIGENCE SUMMARY")
    print("-" * 70)

    lowest_distribution = (
        comparison_df[
            "lowest_marketplace"
        ]
        .value_counts()
    )

    print()
    print("Lowest-price marketplace distribution:")

    for marketplace, count in (
        lowest_distribution.items()
    ):
        print(
            f"  {marketplace}: {count}"
        )

    print()
    print("Top price differences:")

    top_differences = (
        comparison_df
        .sort_values(
            "price_difference",
            ascending=False,
        )
        .head(5)
    )

    for _, row in top_differences.iterrows():

        print(
            f"{row['product_name']} | "
            f"Lowest: {row['lowest_marketplace']} "
            f"₹{row['lowest_price']:,.2f} | "
            f"Highest: ₹{row['highest_price']:,.2f} | "
            f"Difference: ₹{row['price_difference']:,.2f} | "
            f"Savings: "
            f"{row['potential_savings_percent']:.2f}%"
        )

    # Specific iPhone 15 diagnostic.
    iphone_rows = normalized_df[
        normalized_df["product_name"]
        .astype(str)
        .str.contains(
            "iPhone 15",
            case=False,
            na=False,
        )
    ]

    if not iphone_rows.empty:

        print()
        print("iPhone 15 validation:")

        for _, row in iphone_rows.iterrows():

            print(
                f"  {row['marketplace']}: "
                f"₹{row['normalized_price']:,.2f} "
                f"({row['availability']}) "
                f"[{row['scraped_at'].date()}]"
            )

    print()
    print("-" * 70)
    print("NEXT MODULE READY: PRICE HISTORY")
    print("-" * 70)
    print()


# ============================================================
# MAIN
# ============================================================

def main():
    """Run the complete price normalization pipeline."""

    print_header()

    try:
        # 1. Load
        df = load_data()

        # 2. Clean
        df = clean_numeric_columns(df)
        df = clean_dates(df)

        # 3. Validate
        validate_price_data(df)

        # 4. Select latest valid in-stock records
        latest = select_latest_valid_records(df)

        # 5. Calculate normalized prices
        normalized = calculate_normalized_prices(
            latest
        )

        # 6. Create comparison
        comparison = create_comparison(
            normalized
        )

        # 7. Save
        save_outputs(
            normalized,
            comparison,
        )

        # 8. Summary
        display_summary(
            normalized,
            comparison,
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("PRICE NORMALIZATION FAILED")
        print("=" * 70)
        print()
        print(str(error))
        print()

        raise


if __name__ == "__main__":
    main()