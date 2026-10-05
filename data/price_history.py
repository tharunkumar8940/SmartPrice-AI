"""
SmartPrice-AI
Price History Engine

Builds historical price intelligence from the cleaned
marketplace dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"

INPUT_FILE = PROCESSED_DIR / "marketplace_products_clean.csv"

HISTORY_OUTPUT = (
    PROCESSED_DIR / "price_history.csv"
)

SUMMARY_OUTPUT = (
    PROCESSED_DIR / "price_history_summary.csv"
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
    "selling_price",
    "shipping_cost",
    "coupon_discount",
    "effective_price",
    "availability",
    "scraped_at",
]


# ============================================================
# HEADER
# ============================================================

def print_header():
    print()
    print("=" * 70)
    print("SMARTPRICE-AI - PRICE HISTORY ENGINE")
    print("=" * 70)
    print()


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading historical marketplace data...")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Input records: {len(df):,}")

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"- {column}"
                for column in missing
            )
        )

    print("Input validation: PASSED")
    print()

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    numeric_columns = [
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

    df["scraped_at"] = pd.to_datetime(
        df["scraped_at"],
        errors="coerce",
    )

    if df["scraped_at"].isna().any():
        raise ValueError(
            "Invalid scraped_at dates detected."
        )

    # --------------------------------------------------------
    # Historical customer-payable price
    # --------------------------------------------------------

    df["shipping_cost"] = (
        df["shipping_cost"]
        .fillna(0)
        .clip(lower=0)
    )

    df["coupon_discount"] = (
        df["coupon_discount"]
        .fillna(0)
        .clip(lower=0)
    )

    # Only use valid selling prices.
    valid_price = (
        df["selling_price"].notna()
        & (df["selling_price"] > 0)
    )

    df["historical_normalized_price"] = np.where(
        valid_price,
        (
            df["selling_price"]
            + df["shipping_cost"]
            - np.minimum(
                df["coupon_discount"],
                df["selling_price"],
            )
        ),
        np.nan,
    )

    # A historical record is usable for price analysis
    # only when the product had a real selling price.
    df["price_available"] = (
        df["historical_normalized_price"]
        .notna()
        & (
            df["historical_normalized_price"]
            > 0
        )
    )

    return df


# ============================================================
# CREATE DAILY HISTORY
# ============================================================

def create_daily_history(df):

    print("Building daily marketplace price history...")

    history = df[
        df["price_available"]
    ].copy()

    if history.empty:
        raise ValueError(
            "No valid historical price records found."
        )

    history["date"] = (
        history["scraped_at"]
        .dt.date
    )

    history = history[
        [
            "product_id",
            "marketplace",
            "product_name",
            "brand",
            "category",
            "date",
            "selling_price",
            "shipping_cost",
            "coupon_discount",
            "historical_normalized_price",
            "availability",
        ]
    ]

    history = history.sort_values(
        by=[
            "product_id",
            "marketplace",
            "date",
        ]
    )

    history = history.rename(
        columns={
            "historical_normalized_price":
                "normalized_price"
        }
    )

    # --------------------------------------------------------
    # Daily price movement
    # --------------------------------------------------------

    history["previous_price"] = (
        history
        .groupby(
            [
                "product_id",
                "marketplace",
            ]
        )["normalized_price"]
        .shift(1)
    )

    history["price_change"] = (
        history["normalized_price"]
        - history["previous_price"]
    )

    history["price_change_percent"] = np.where(
        history["previous_price"].notna()
        & (history["previous_price"] > 0),
        (
            history["price_change"]
            / history["previous_price"]
        )
        * 100,
        np.nan,
    )

    # --------------------------------------------------------
    # Running historical minimum / maximum
    # --------------------------------------------------------

    history["historical_low"] = (
        history
        .groupby(
            [
                "product_id",
                "marketplace",
            ]
        )["normalized_price"]
        .cummin()
    )

    history["historical_high"] = (
        history
        .groupby(
            [
                "product_id",
                "marketplace",
            ]
        )["normalized_price"]
        .cummax()
    )

    # --------------------------------------------------------
    # Trend classification
    # --------------------------------------------------------

    history["trend"] = np.select(
        [
            history["price_change_percent"] > 2,
            history["price_change_percent"] < -2,
        ],
        [
            "Increasing",
            "Decreasing",
        ],
        default="Stable",
    )

    history = history.reset_index(
        drop=True
    )

    return history


# ============================================================
# CREATE PRODUCT SUMMARY
# ============================================================

def create_summary(history):

    print("Creating product price intelligence summary...")

    summary_rows = []

    grouped = history.groupby(
        [
            "product_id",
            "marketplace",
        ],
        sort=True,
    )

    for (
        product_id,
        marketplace,
    ), group in grouped:

        group = group.sort_values(
            "date"
        )

        prices = group[
            "normalized_price"
        ].dropna()

        if prices.empty:
            continue

        current_row = group.iloc[-1]

        current_price = float(
            current_row[
                "normalized_price"
            ]
        )

        minimum_price = float(
            prices.min()
        )

        maximum_price = float(
            prices.max()
        )

        average_price = float(
            prices.mean()
        )

        median_price = float(
            prices.median()
        )

        price_std = float(
            prices.std()
        ) if len(prices) > 1 else 0.0

        first_price = float(
            prices.iloc[0]
        )

        total_change = (
            current_price
            - first_price
        )

        total_change_percent = (
            (
                total_change
                / first_price
            )
            * 100
            if first_price > 0
            else 0
        )

        # Distance from historical low.
        above_low_percent = (
            (
                (
                    current_price
                    - minimum_price
                )
                / minimum_price
            )
            * 100
            if minimum_price > 0
            else 0
        )

        # Distance from historical high.
        below_high_percent = (
            (
                (
                    maximum_price
                    - current_price
                )
                / maximum_price
            )
            * 100
            if maximum_price > 0
            else 0
        )

        # Best historical saving opportunity.
        historical_savings = (
            current_price
            - minimum_price
        )

        if total_change_percent > 2:
            overall_trend = "Increasing"

        elif total_change_percent < -2:
            overall_trend = "Decreasing"

        else:
            overall_trend = "Stable"

        # Deal classification.
        if (
            current_price
            <= minimum_price * 1.02
        ):
            deal_status = "Excellent Deal"

        elif (
            current_price
            <= minimum_price * 1.05
        ):
            deal_status = "Good Deal"

        elif (
            current_price
            <= average_price
        ):
            deal_status = "Fair Price"

        else:
            deal_status = "Above Average"

        summary_rows.append(
            {
                "product_id": product_id,
                "marketplace": marketplace,
                "product_name":
                    group["product_name"].iloc[0],
                "brand":
                    group["brand"].iloc[0],
                "category":
                    group["category"].iloc[0],
                "history_start":
                    group["date"].min(),
                "history_end":
                    group["date"].max(),
                "observations":
                    len(group),
                "current_price":
                    round(
                        current_price,
                        2,
                    ),
                "minimum_price":
                    round(
                        minimum_price,
                        2,
                    ),
                "maximum_price":
                    round(
                        maximum_price,
                        2,
                    ),
                "average_price":
                    round(
                        average_price,
                        2,
                    ),
                "median_price":
                    round(
                        median_price,
                        2,
                    ),
                "price_volatility":
                    round(
                        price_std,
                        2,
                    ),
                "total_price_change":
                    round(
                        total_change,
                        2,
                    ),
                "total_price_change_percent":
                    round(
                        total_change_percent,
                        2,
                    ),
                "above_historical_low_percent":
                    round(
                        above_low_percent,
                        2,
                    ),
                "below_historical_high_percent":
                    round(
                        below_high_percent,
                        2,
                    ),
                "historical_savings_opportunity":
                    round(
                        historical_savings,
                        2,
                    ),
                "overall_trend":
                    overall_trend,
                "deal_status":
                    deal_status,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    if summary.empty:
        raise ValueError(
            "No historical price summaries created."
        )

    return summary


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    history,
    summary,
):

    history.to_csv(
        HISTORY_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print()
    print("PRICE HISTORY COMPLETE")
    print()

    print("Historical file:")
    print(HISTORY_OUTPUT)
    print()

    print("Summary file:")
    print(SUMMARY_OUTPUT)
    print()

    print(
        f"Historical price records: "
        f"{len(history):,}"
    )

    print(
        f"Marketplace summaries: "
        f"{len(summary):,}"
    )


# ============================================================
# DISPLAY SUMMARY
# ============================================================

def display_summary(
    history,
    summary,
):

    print()
    print("-" * 70)
    print("PRICE HISTORY INTELLIGENCE")
    print("-" * 70)

    print()
    print("Overall trend distribution:")

    trend_counts = (
        summary["overall_trend"]
        .value_counts()
    )

    for trend, count in (
        trend_counts.items()
    ):
        print(
            f"  {trend}: {count}"
        )

    print()
    print("Deal status distribution:")

    deal_counts = (
        summary["deal_status"]
        .value_counts()
    )

    for status, count in (
        deal_counts.items()
    ):
        print(
            f"  {status}: {count}"
        )

    print()
    print("Top current deals:")

    top_deals = (
        summary
        .sort_values(
            "above_historical_low_percent"
        )
        .head(5)
    )

    for _, row in top_deals.iterrows():

        print(
            f"{row['product_name']} | "
            f"{row['marketplace']} | "
            f"Current: ₹"
            f"{row['current_price']:,.2f} | "
            f"Historical Low: ₹"
            f"{row['minimum_price']:,.2f} | "
            f"Status: "
            f"{row['deal_status']}"
        )

    print()
    print("Highest price volatility:")

    volatile = (
        summary
        .sort_values(
            "price_volatility",
            ascending=False,
        )
        .head(5)
    )

    for _, row in volatile.iterrows():

        print(
            f"{row['product_name']} | "
            f"{row['marketplace']} | "
            f"Volatility: ₹"
            f"{row['price_volatility']:,.2f}"
        )

    print()
    print("-" * 70)
    print("NEXT MODULE READY: DEAL DETECTION")
    print("-" * 70)
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print_header()

    try:

        # 1. Load
        df = load_data()

        # 2. Prepare
        df = prepare_data(df)

        # 3. Historical daily records
        history = create_daily_history(
            df
        )

        # 4. Product-marketplace summaries
        summary = create_summary(
            history
        )

        # 5. Save
        save_outputs(
            history,
            summary,
        )

        # 6. Display
        display_summary(
            history,
            summary,
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("PRICE HISTORY FAILED")
        print("=" * 70)
        print()
        print(str(error))
        print()

        raise


if __name__ == "__main__":
    main()