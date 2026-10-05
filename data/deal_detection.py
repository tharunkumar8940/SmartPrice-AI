"""
SmartPrice-AI
Deal Detection Engine

Detects genuine marketplace deals using:
    - Current price
    - Historical minimum
    - Historical average
    - Historical maximum
    - Price volatility
    - Price movement
    - Marketplace comparison
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"

SUMMARY_FILE = (
    PROCESSED_DIR / "price_history_summary.csv"
)

COMPARISON_FILE = (
    PROCESSED_DIR / "marketplace_price_comparison.csv"
)

DEAL_OUTPUT = (
    PROCESSED_DIR / "deal_detection_results.csv"
)


# ============================================================
# HEADER
# ============================================================

def print_header():

    print()
    print("=" * 70)
    print("SMARTPRICE-AI - DEAL DETECTION ENGINE")
    print("=" * 70)
    print()


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading price intelligence data...")

    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Price history summary not found:\n"
            f"{SUMMARY_FILE}\n\n"
            f"Run price_history.py first."
        )

    if not COMPARISON_FILE.exists():
        raise FileNotFoundError(
            f"Marketplace comparison not found:\n"
            f"{COMPARISON_FILE}\n\n"
            f"Run price_normalization.py first."
        )

    summary = pd.read_csv(
        SUMMARY_FILE
    )

    comparison = pd.read_csv(
        COMPARISON_FILE
    )

    print(
        f"Price history summaries: "
        f"{len(summary):,}"
    )

    print(
        f"Marketplace comparisons: "
        f"{len(comparison):,}"
    )

    print("Input validation: PASSED")
    print()

    return summary, comparison


# ============================================================
# CALCULATE DEAL SCORE
# ============================================================

def calculate_deal_score(row):
    """
    Calculate a 0-100 deal score.

    Components:

        40% historical price position
        30% marketplace savings
        20% historical trend
        10% volatility opportunity
    """

    # --------------------------------------------------------
    # Historical price position
    # --------------------------------------------------------

    current_price = row[
        "current_price"
    ]

    historical_low = row[
        "minimum_price"
    ]

    historical_high = row[
        "maximum_price"
    ]

    if historical_high > historical_low:

        historical_position = (
            (
                historical_high
                - current_price
            )
            /
            (
                historical_high
                - historical_low
            )
        ) * 100

    else:

        historical_position = 100

    historical_position = np.clip(
        historical_position,
        0,
        100,
    )

    # --------------------------------------------------------
    # Marketplace advantage
    # --------------------------------------------------------

    marketplace_savings = row[
        "marketplace_savings_percent"
    ]

    marketplace_score = np.clip(
        marketplace_savings * 10,
        0,
        100,
    )

    # --------------------------------------------------------
    # Trend score
    # --------------------------------------------------------

    trend = row[
        "overall_trend"
    ]

    if trend == "Decreasing":

        trend_score = 100

    elif trend == "Stable":

        trend_score = 60

    else:

        trend_score = 20

    # --------------------------------------------------------
    # Volatility opportunity
    # --------------------------------------------------------

    average_price = row[
        "average_price"
    ]

    volatility = row[
        "price_volatility"
    ]

    if average_price > 0:

        volatility_percent = (
            volatility
            / average_price
        ) * 100

    else:

        volatility_percent = 0

    volatility_score = np.clip(
        volatility_percent * 10,
        0,
        100,
    )

    # --------------------------------------------------------
    # Final weighted score
    # --------------------------------------------------------

    score = (
        historical_position * 0.40
        +
        marketplace_score * 0.30
        +
        trend_score * 0.20
        +
        volatility_score * 0.10
    )

    return round(
        float(
            np.clip(
                score,
                0,
                100,
            )
        ),
        2,
    )


# ============================================================
# CLASSIFY DEAL
# ============================================================

def classify_deal(score):

    if score >= 85:

        return "Exceptional Deal"

    if score >= 70:

        return "Strong Deal"

    if score >= 55:

        return "Good Deal"

    if score >= 40:

        return "Fair Deal"

    return "Not a Deal"


# ============================================================
# BUILD DEAL DATASET
# ============================================================

def build_deal_dataset(
    summary,
    comparison,
):

    print("Calculating deal intelligence...")

    rows = []

    for _, row in summary.iterrows():

        product_id = row[
            "product_id"
        ]

        product_comparison = comparison[
            comparison[
                "product_id"
            ]
            == product_id
        ]

        if product_comparison.empty:

            continue

        comparison_row = (
            product_comparison.iloc[0]
        )

        current_price = float(
            row["current_price"]
        )

        lowest_price = float(
            comparison_row[
                "lowest_price"
            ]
        )

        highest_price = float(
            comparison_row[
                "highest_price"
            ]
        )

        # ----------------------------------------------------
        # Marketplace advantage
        # ----------------------------------------------------

        if current_price > 0:

            marketplace_savings_percent = (
                (
                    current_price
                    - lowest_price
                )
                /
                current_price
            ) * 100

        else:

            marketplace_savings_percent = 0

        # ----------------------------------------------------
        # Historical discount
        # ----------------------------------------------------

        if (
            row["maximum_price"]
            > current_price
        ):

            historical_discount_percent = (
                (
                    row["maximum_price"]
                    - current_price
                )
                /
                row["maximum_price"]
            ) * 100

        else:

            historical_discount_percent = 0

        # ----------------------------------------------------
        # Distance from historical low
        # ----------------------------------------------------

        if row["minimum_price"] > 0:

            distance_from_low_percent = (
                (
                    current_price
                    - row["minimum_price"]
                )
                /
                row["minimum_price"]
            ) * 100

        else:

            distance_from_low_percent = 0

        # ----------------------------------------------------
        # Deal score
        # ----------------------------------------------------

        temp_row = row.copy()

        temp_row[
            "marketplace_savings_percent"
        ] = marketplace_savings_percent

        deal_score = calculate_deal_score(
            temp_row
        )

        deal_category = classify_deal(
            deal_score
        )

        # ----------------------------------------------------
        # Recommendation
        # ----------------------------------------------------

        if deal_score >= 85:

            recommendation = (
                "BUY NOW - Exceptional deal"
            )

        elif deal_score >= 70:

            recommendation = (
                "BUY - Strong deal"
            )

        elif deal_score >= 55:

            recommendation = (
                "CONSIDER - Good deal"
            )

        elif deal_score >= 40:

            recommendation = (
                "WAIT - Fair price"
            )

        else:

            recommendation = (
                "WAIT - Not a strong deal"
            )

        rows.append(
            {
                "product_id":
                    product_id,

                "product_name":
                    row["product_name"],

                "brand":
                    row["brand"],

                "category":
                    row["category"],

                "marketplace":
                    row["marketplace"],

                "current_price":
                    round(
                        current_price,
                        2,
                    ),

                "historical_low":
                    round(
                        row["minimum_price"],
                        2,
                    ),

                "historical_average":
                    round(
                        row["average_price"],
                        2,
                    ),

                "historical_high":
                    round(
                        row["maximum_price"],
                        2,
                    ),

                "lowest_marketplace_price":
                    round(
                        lowest_price,
                        2,
                    ),

                "highest_marketplace_price":
                    round(
                        highest_price,
                        2,
                    ),

                "marketplace_savings_percent":
                    round(
                        marketplace_savings_percent,
                        2,
                    ),

                "historical_discount_percent":
                    round(
                        historical_discount_percent,
                        2,
                    ),

                "distance_from_historical_low_percent":
                    round(
                        distance_from_low_percent,
                        2,
                    ),

                "price_volatility":
                    round(
                        row["price_volatility"],
                        2,
                    ),

                "overall_trend":
                    row["overall_trend"],

                "deal_score":
                    deal_score,

                "deal_category":
                    deal_category,

                "recommendation":
                    recommendation,

                "deal_status":
                    row["deal_status"],
            }
        )

    deals = pd.DataFrame(
        rows
    )

    if deals.empty:

        raise ValueError(
            "No deal detection records created."
        )

    return deals


# ============================================================
# SAVE
# ============================================================

def save_results(deals):

    deals = deals.sort_values(
        by="deal_score",
        ascending=False,
    ).reset_index(
        drop=True
    )

    deals.to_csv(
        DEAL_OUTPUT,
        index=False,
    )

    print()
    print("DEAL DETECTION COMPLETE")
    print()

    print("Output file:")
    print(DEAL_OUTPUT)

    print()
    print(
        f"Deal records: "
        f"{len(deals):,}"
    )

    return deals


# ============================================================
# DISPLAY
# ============================================================

def display_results(deals):

    print()
    print("-" * 70)
    print("DEAL INTELLIGENCE SUMMARY")
    print("-" * 70)

    print()
    print("Deal category distribution:")

    category_counts = (
        deals[
            "deal_category"
        ]
        .value_counts()
    )

    for category, count in (
        category_counts.items()
    ):

        print(
            f"  {category}: {count}"
        )

    print()
    print("Top detected deals:")

    top_deals = (
        deals
        .sort_values(
            "deal_score",
            ascending=False,
        )
        .head(10)
    )

    for _, row in top_deals.iterrows():

        print(
            f"{row['product_name']} | "
            f"{row['marketplace']} | "
            f"₹{row['current_price']:,.2f} | "
            f"Score: {row['deal_score']:.2f} | "
            f"{row['deal_category']}"
        )

    print()
    print("BUY NOW recommendations:")

    buy_now = deals[
        deals["deal_score"] >= 85
    ].head(5)

    if buy_now.empty:

        print("  No exceptional deals detected.")

    else:

        for _, row in buy_now.iterrows():

            print(
                f"  {row['product_name']} | "
                f"{row['marketplace']} | "
                f"₹{row['current_price']:,.2f}"
            )

    print()
    print("-" * 70)
    print("NEXT MODULE READY: SMART RECOMMENDATION")
    print("-" * 70)
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print_header()

    try:

        summary, comparison = (
            load_data()
        )

        deals = build_deal_dataset(
            summary,
            comparison,
        )

        deals = save_results(
            deals
        )

        display_results(
            deals
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("DEAL DETECTION FAILED")
        print("=" * 70)
        print()
        print(str(error))
        print()

        raise


if __name__ == "__main__":
    main()