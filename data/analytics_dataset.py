"""
SMARTPRICE-AI
Analytics Dataset Builder

Creates the unified analytics dataset used by the Streamlit dashboard.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

DEAL_FILE = PROCESSED_DIR / "deal_detection_results.csv"
PRICE_FILE = PROCESSED_DIR / "normalized_marketplace_prices.csv"
HISTORY_FILE = PROCESSED_DIR / "price_history_summary.csv"
RECOMMENDATION_FILE = PROCESSED_DIR / "smart_recommendations.csv"

OUTPUT_FILE = PROCESSED_DIR / "analytics_dataset.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(path: Path, name: str) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"{name} file not found:\n{path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"{name} file is empty:\n{path}"
        )

    return df


def find_column(df: pd.DataFrame, candidates):

    for column in candidates:
        if column in df.columns:
            return column

    return None


def clean_numeric(df: pd.DataFrame, columns):

    for column in columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SMARTPRICE-AI - ANALYTICS DATASET BUILDER")
print("=" * 70)
print()


# ============================================================
# LOAD DATA
# ============================================================

print("Loading intelligence datasets...")

deal_df = load_csv(
    DEAL_FILE,
    "Deal detection"
)

price_df = load_csv(
    PRICE_FILE,
    "Normalized price"
)

history_df = load_csv(
    HISTORY_FILE,
    "Price history"
)

recommendation_df = load_csv(
    RECOMMENDATION_FILE,
    "Smart recommendation"
)

print(f"Deal records: {len(deal_df):,}")
print(f"Price records: {len(price_df):,}")
print(f"History records: {len(history_df):,}")
print(f"Recommendation records: {len(recommendation_df):,}")
print()


# ============================================================
# STANDARDIZE KEY COLUMNS
# ============================================================

product_id_col = find_column(
    price_df,
    ["product_id", "Product ID"]
)

product_name_col = find_column(
    price_df,
    ["product_name", "Product Name"]
)

marketplace_col = find_column(
    price_df,
    ["marketplace", "Marketplace"]
)

if product_id_col is None:
    raise ValueError(
        "Product ID column not found."
    )

if product_name_col is None:
    raise ValueError(
        "Product name column not found."
    )

if marketplace_col is None:
    raise ValueError(
        "Marketplace column not found."
    )


price_df = price_df.rename(
    columns={
        product_id_col: "product_id",
        product_name_col: "product_name",
        marketplace_col: "marketplace",
    }
)


# ============================================================
# STANDARDIZE OTHER DATASETS
# ============================================================

def standardize_keys(df, name):

    product_id = find_column(
        df,
        ["product_id", "Product ID"]
    )

    marketplace = find_column(
        df,
        ["marketplace", "Marketplace"]
    )

    if product_id is None or marketplace is None:

        raise ValueError(
            f"{name} data must contain product_id "
            f"and marketplace columns."
        )

    return df.rename(
        columns={
            product_id: "product_id",
            marketplace: "marketplace",
        }
    )


deal_df = standardize_keys(
    deal_df,
    "Deal detection"
)

history_df = standardize_keys(
    history_df,
    "Price history"
)

recommendation_df = standardize_keys(
    recommendation_df,
    "Smart recommendation"
)


# ============================================================
# CLEAN NUMERIC DATA
# ============================================================

print("Cleaning analytics inputs...")

price_df = clean_numeric(
    price_df,
    [
        "selling_price",
        "effective_price",
        "original_price",
        "shipping_cost",
        "coupon_discount",
        "discount_percent",
        "rating",
        "review_count",
    ]
)

deal_df = clean_numeric(
    deal_df,
    [
        "current_price",
        "deal_score",
        "historical_price_score",
        "marketplace_savings_score",
        "trend_score",
        "volatility_score",
    ]
)

history_df = clean_numeric(
    history_df,
    [
        "historical_min",
        "historical_max",
        "historical_average",
        "historical_median",
        "price_volatility",
        "historical_savings_opportunity",
    ]
)

recommendation_df = clean_numeric(
    recommendation_df,
    [
        "current_price",
        "recommendation_score",
        "deal_score",
        "rating_score",
        "historical_price_score",
        "marketplace_advantage_score",
    ]
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

price_df = price_df.drop_duplicates(
    subset=["product_id", "marketplace"],
    keep="last"
)

deal_df = deal_df.drop_duplicates(
    subset=["product_id", "marketplace"],
    keep="last"
)

history_df = history_df.drop_duplicates(
    subset=["product_id", "marketplace"],
    keep="last"
)

recommendation_df = recommendation_df.drop_duplicates(
    subset=["product_id", "marketplace"],
    keep="last"
)


# ============================================================
# BASE ANALYTICS DATASET
# ============================================================

analytics_df = price_df.copy()


# ============================================================
# DEAL INTELLIGENCE
# ============================================================

deal_columns = [
    "product_id",
    "marketplace",
    "deal_score",
    "deal_category",
    "current_price",
    "historical_price_score",
    "marketplace_savings_score",
    "trend_score",
    "volatility_score",
]

deal_columns = [
    column
    for column in deal_columns
    if column in deal_df.columns
]

analytics_df = analytics_df.merge(
    deal_df[deal_columns],
    on=["product_id", "marketplace"],
    how="left",
    suffixes=("", "_deal")
)


# ============================================================
# PRICE HISTORY INTELLIGENCE
# ============================================================

history_columns = [
    "product_id",
    "marketplace",
    "historical_min",
    "historical_max",
    "historical_average",
    "historical_median",
    "price_volatility",
    "historical_savings_opportunity",
    "overall_trend",
    "deal_status",
]

history_columns = [
    column
    for column in history_columns
    if column in history_df.columns
]

analytics_df = analytics_df.merge(
    history_df[history_columns],
    on=["product_id", "marketplace"],
    how="left",
    suffixes=("", "_history")
)


# ============================================================
# SMART RECOMMENDATION INTELLIGENCE
# ============================================================

recommendation_columns = [
    "product_id",
    "marketplace",
    "recommendation_score",
    "recommendation",
    "recommendation_reason",
    "action",
    "rating_score",
    "marketplace_advantage_score",
]

recommendation_columns = [
    column
    for column in recommendation_columns
    if column in recommendation_df.columns
]

analytics_df = analytics_df.merge(
    recommendation_df[recommendation_columns],
    on=["product_id", "marketplace"],
    how="left",
    suffixes=("", "_recommendation")
)


# ============================================================
# REMOVE POSSIBLE DUPLICATE METRIC COLUMNS
# ============================================================

# If earlier modules already supplied these metrics,
# retain the base version and remove duplicate merge versions.

duplicate_metric_columns = [
    "deal_score_deal",
    "current_price_deal",
    "historical_price_score_deal",
    "rating_score_recommendation",
    "marketplace_advantage_score_recommendation",
]

for column in duplicate_metric_columns:

    if column in analytics_df.columns:

        base_column = column.replace(
            "_deal",
            ""
        ).replace(
            "_recommendation",
            ""
        )

        if base_column not in analytics_df.columns:

            analytics_df = analytics_df.rename(
                columns={
                    column: base_column
                }
            )

        else:

            analytics_df = analytics_df.drop(
                columns=[column]
            )


# ============================================================
# MARKETPLACE PRICE STATISTICS
# ============================================================

print("Calculating marketplace rankings...")

if "effective_price" not in analytics_df.columns:

    raise ValueError(
        "effective_price column is missing."
    )


analytics_df["effective_price"] = pd.to_numeric(
    analytics_df["effective_price"],
    errors="coerce"
)


marketplace_stats = (
    analytics_df
    .groupby("product_id")
    .agg(
        lowest_marketplace_price=(
            "effective_price",
            "min"
        ),
        highest_marketplace_price=(
            "effective_price",
            "max"
        ),
        average_marketplace_price=(
            "effective_price",
            "mean"
        ),
        marketplace_count=(
            "marketplace",
            "nunique"
        ),
    )
    .reset_index()
)


marketplace_stats[
    "marketplace_price_difference"
] = (
    marketplace_stats[
        "highest_marketplace_price"
    ]
    - marketplace_stats[
        "lowest_marketplace_price"
    ]
)


marketplace_stats[
    "marketplace_savings_percent"
] = np.where(
    marketplace_stats[
        "highest_marketplace_price"
    ] > 0,

    (
        marketplace_stats[
            "marketplace_price_difference"
        ]
        /
        marketplace_stats[
            "highest_marketplace_price"
        ]
    ) * 100,

    0,
)


# ============================================================
# IMPORTANT:
# ONLY CREATE THE PRODUCT-LEVEL LOWEST PRICE HERE
# ============================================================

analytics_df = analytics_df.drop(
    columns=[
        column
        for column in [
            "lowest_marketplace_price",
            "highest_marketplace_price",
            "average_marketplace_price",
            "marketplace_count",
            "marketplace_price_difference",
            "marketplace_savings_percent",
        ]
        if column in analytics_df.columns
    ],
    errors="ignore"
)


analytics_df = analytics_df.merge(
    marketplace_stats,
    on="product_id",
    how="left"
)


# ============================================================
# MARKETPLACE RANK
# ============================================================

analytics_df["marketplace_price_rank"] = (
    analytics_df
    .groupby("product_id")["effective_price"]
    .rank(
        method="min",
        ascending=True
    )
)


analytics_df["is_lowest_price"] = (
    analytics_df["marketplace_price_rank"] == 1
)


# ============================================================
# PRICE POSITION
# ============================================================

print("Calculating price intelligence metrics...")

if "historical_min" in analytics_df.columns:

    analytics_df[
        "distance_from_historical_low"
    ] = (
        analytics_df["effective_price"]
        - analytics_df["historical_min"]
    )

else:

    analytics_df[
        "distance_from_historical_low"
    ] = np.nan


if "historical_max" in analytics_df.columns:

    analytics_df[
        "distance_from_historical_high"
    ] = (
        analytics_df["historical_max"]
        - analytics_df["effective_price"]
    )

else:

    analytics_df[
        "distance_from_historical_high"
    ] = np.nan


if (
    "historical_min" in analytics_df.columns
    and
    "historical_max" in analytics_df.columns
):

    price_range = (
        analytics_df["historical_max"]
        - analytics_df["historical_min"]
    )

    analytics_df[
        "historical_price_position"
    ] = np.where(
        price_range > 0,

        (
            (
                analytics_df["effective_price"]
                - analytics_df["historical_min"]
            )
            /
            price_range
        ) * 100,

        50,
    )

else:

    analytics_df[
        "historical_price_position"
    ] = 50


# ============================================================
# MARKETPLACE SAVINGS
# ============================================================

analytics_df[
    "savings_vs_highest_marketplace"
] = np.maximum(
    analytics_df[
        "effective_price"
    ]
    -
    analytics_df[
        "lowest_marketplace_price"
    ],
    0,
)


analytics_df[
    "savings_vs_highest_percent"
] = np.where(
    analytics_df["effective_price"] > 0,

    (
        analytics_df[
            "savings_vs_highest_marketplace"
        ]
        /
        analytics_df[
            "effective_price"
        ]
    ) * 100,

    0,
)


# ============================================================
# PRODUCT / MARKETPLACE CLEANUP
# ============================================================

analytics_df["product_name"] = (
    analytics_df["product_name"]
    .astype(str)
    .str.strip()
)

analytics_df["marketplace"] = (
    analytics_df["marketplace"]
    .astype(str)
    .str.strip()
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("Running final analytics validation...")

required_columns = [
    "product_id",
    "product_name",
    "marketplace",
    "effective_price",
    "lowest_marketplace_price",
    "highest_marketplace_price",
    "marketplace_price_rank",
    "is_lowest_price",
]

missing_columns = [
    column
    for column in required_columns
    if column not in analytics_df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required analytics columns: "
        + str(missing_columns)
    )


if analytics_df.empty:

    raise ValueError(
        "Analytics dataset is empty."
    )


analytics_df = analytics_df.drop_duplicates(
    subset=[
        "product_id",
        "marketplace",
    ],
    keep="first"
)


# ============================================================
# SORT
# ============================================================

analytics_df = analytics_df.sort_values(
    by=[
        "product_id",
        "effective_price",
    ],
    ascending=[
        True,
        True,
    ]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

analytics_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("ANALYTICS DATASET COMPLETE")
print("-" * 70)

print(
    f"Output file:\n{OUTPUT_FILE}"
)

print(
    f"\nAnalytics records: "
    f"{len(analytics_df):,}"
)

print(
    f"Analytics columns: "
    f"{len(analytics_df.columns):,}"
)

print(
    f"Products: "
    f"{analytics_df['product_id'].nunique():,}"
)

print(
    f"Marketplaces: "
    f"{analytics_df['marketplace'].nunique():,}"
)


print()
print("Marketplace distribution:")

for marketplace, count in (
    analytics_df["marketplace"]
    .value_counts()
    .items()
):

    print(
        f"  {marketplace}: {count}"
    )


print()
print("Lowest-price marketplace distribution:")

lowest_prices = analytics_df[
    analytics_df["is_lowest_price"]
]

for marketplace, count in (
    lowest_prices["marketplace"]
    .value_counts()
    .items()
):

    print(
        f"  {marketplace}: {count}"
    )


if "recommendation" in analytics_df.columns:

    print()
    print("Recommendation distribution:")

    for recommendation, count in (
        analytics_df["recommendation"]
        .value_counts()
        .items()
    ):

        print(
            f"  {recommendation}: {count}"
        )


if "deal_category" in analytics_df.columns:

    print()
    print("Deal category distribution:")

    for category, count in (
        analytics_df["deal_category"]
        .value_counts()
        .items()
    ):

        print(
            f"  {category}: {count}"
        )


print()
print("Dataset preview:")
print(
    analytics_df[
        [
            "product_id",
            "product_name",
            "marketplace",
            "effective_price",
            "lowest_marketplace_price",
            "marketplace_price_rank",
            "is_lowest_price",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print()
print("=" * 70)
print("NEXT MODULE READY: STREAMLIT ANALYTICS DASHBOARD")
print("=" * 70)