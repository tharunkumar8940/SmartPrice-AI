"""
SmartPrice-AI
Smart Recommendation Engine

Combines price intelligence, historical trends,
deal scores, ratings, and marketplace prices
to generate shopping recommendations.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"

DEAL_FILE = (
    PROCESSED_DIR / "deal_detection_results.csv"
)

NORMALIZED_FILE = (
    PROCESSED_DIR / "normalized_marketplace_prices.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR / "smart_recommendations.csv"
)


# ============================================================
# HEADER
# ============================================================

def print_header():

    print()
    print("=" * 70)
    print("SMARTPRICE-AI - SMART RECOMMENDATION ENGINE")
    print("=" * 70)
    print()


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading deal intelligence...")

    if not DEAL_FILE.exists():
        raise FileNotFoundError(
            f"Deal detection file not found:\n"
            f"{DEAL_FILE}\n\n"
            f"Run deal_detection.py first."
        )

    if not NORMALIZED_FILE.exists():
        raise FileNotFoundError(
            f"Normalized price file not found:\n"
            f"{NORMALIZED_FILE}\n\n"
            f"Run price_normalization.py first."
        )

    deals = pd.read_csv(
        DEAL_FILE
    )

    prices = pd.read_csv(
        NORMALIZED_FILE
    )

    print(
        f"Deal records: {len(deals):,}"
    )

    print(
        f"Current price records: "
        f"{len(prices):,}"
    )

    print("Input validation: PASSED")
    print()

    return deals, prices


# ============================================================
# CALCULATE RECOMMENDATION SCORE
# ============================================================

def calculate_recommendation_score(row):

    deal_score = float(
        row["deal_score"]
    )

    rating = float(
        row["rating"]
    ) if pd.notna(
        row["rating"]
    ) else 0

    # Convert rating to 0-100.
    rating_score = (
        rating / 5
    ) * 100

    historical_position = float(
        row[
            "historical_price_score"
        ]
    )

    marketplace_advantage = float(
        row[
            "marketplace_advantage_score"
        ]
    )

    # --------------------------------------------------------
    # Weighted recommendation model
    # --------------------------------------------------------

    score = (
        deal_score * 0.45
        +
        rating_score * 0.20
        +
        historical_position * 0.20
        +
        marketplace_advantage * 0.15
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
# RECOMMENDATION CATEGORY
# ============================================================

def classify_recommendation(score):

    if score >= 80:

        return "BUY NOW"

    if score >= 65:

        return "BUY"

    if score >= 50:

        return "CONSIDER"

    return "WAIT"


# ============================================================
# BUILD RECOMMENDATIONS
# ============================================================

def build_recommendations(
    deals,
    prices,
):

    print(
        "Generating personalized shopping recommendations..."
    )

    rows = []

    for _, deal in deals.iterrows():

        product_id = deal[
            "product_id"
        ]

        marketplace = deal[
            "marketplace"
        ]

        price_rows = prices[
            (prices["product_id"] == product_id)
            &
            (
                prices["marketplace"]
                == marketplace
            )
        ]

        if price_rows.empty:

            continue

        price_row = price_rows.iloc[0]

        current_price = float(
            deal["current_price"]
        )

        historical_low = float(
            deal["historical_low"]
        )

        historical_average = float(
            deal["historical_average"]
        )

        historical_high = float(
            deal["historical_high"]
        )

        rating = (
            float(
                price_row["rating"]
            )
            if pd.notna(
                price_row["rating"]
            )
            else 0
        )

        # ----------------------------------------------------
        # Historical price score
        # ----------------------------------------------------

        if (
            historical_high
            > historical_low
        ):

            historical_price_score = (
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

            historical_price_score = 100

        historical_price_score = np.clip(
            historical_price_score,
            0,
            100,
        )

        # ----------------------------------------------------
        # Marketplace advantage
        # ----------------------------------------------------

        lowest_marketplace_price = float(
            deal[
                "lowest_marketplace_price"
            ]
        )

        if current_price > 0:

            marketplace_advantage = (
                (
                    current_price
                    - lowest_marketplace_price
                )
                /
                current_price
            ) * 100

        else:

            marketplace_advantage = 0

        marketplace_advantage_score = np.clip(
            marketplace_advantage * 10,
            0,
            100,
        )

        # ----------------------------------------------------
        # Recommendation score
        # ----------------------------------------------------

        recommendation_row = deal.copy()

        recommendation_row[
            "rating"
        ] = rating

        recommendation_row[
            "historical_price_score"
        ] = historical_price_score

        recommendation_row[
            "marketplace_advantage_score"
        ] = marketplace_advantage_score

        recommendation_score = (
            calculate_recommendation_score(
                recommendation_row
            )
        )

        recommendation = (
            classify_recommendation(
                recommendation_score
            )
        )

        # ----------------------------------------------------
        # Explanation
        # ----------------------------------------------------

        reasons = []

        if current_price <= historical_low * 1.02:

            reasons.append(
                "Near historical low"
            )

        elif current_price <= historical_average:

            reasons.append(
                "Below historical average"
            )

        if (
            deal["overall_trend"]
            == "Decreasing"
        ):

            reasons.append(
                "Price trend is decreasing"
            )

        elif (
            deal["overall_trend"]
            == "Increasing"
        ):

            reasons.append(
                "Price trend is increasing"
            )

        if (
            current_price
            == lowest_marketplace_price
        ):

            reasons.append(
                "Lowest marketplace price"
            )

        elif marketplace_advantage > 2:

            reasons.append(
                "Competitive marketplace price"
            )

        if rating >= 4.5:

            reasons.append(
                "Excellent customer rating"
            )

        elif rating >= 4.0:

            reasons.append(
                "Strong customer rating"
            )

        if not reasons:

            reasons.append(
                "Price requires further monitoring"
            )

        explanation = "; ".join(
            reasons
        )

        # ----------------------------------------------------
        # Final recommendation text
        # ----------------------------------------------------

        if recommendation == "BUY NOW":

            action = (
                "Strong buying opportunity. "
                "Current price and market conditions "
                "support purchasing now."
            )

        elif recommendation == "BUY":

            action = (
                "Good buying opportunity with "
                "favorable price intelligence."
            )

        elif recommendation == "CONSIDER":

            action = (
                "Reasonable price. Compare alternatives "
                "or monitor the price before purchasing."
            )

        else:

            action = (
                "Current conditions suggest waiting "
                "for a better price."
            )

        rows.append(
            {
                "product_id":
                    product_id,

                "product_name":
                    deal["product_name"],

                "brand":
                    deal["brand"],

                "category":
                    deal["category"],

                "marketplace":
                    marketplace,

                "current_price":
                    round(
                        current_price,
                        2,
                    ),

                "historical_low":
                    round(
                        historical_low,
                        2,
                    ),

                "historical_average":
                    round(
                        historical_average,
                        2,
                    ),

                "historical_high":
                    round(
                        historical_high,
                        2,
                    ),

                "rating":
                    round(
                        rating,
                        2,
                    ),

                "deal_score":
                    round(
                        deal["deal_score"],
                        2,
                    ),

                "historical_price_score":
                    round(
                        historical_price_score,
                        2,
                    ),

                "marketplace_advantage_score":
                    round(
                        marketplace_advantage_score,
                        2,
                    ),

                "recommendation_score":
                    recommendation_score,

                "recommendation":
                    recommendation,

                "price_trend":
                    deal["overall_trend"],

                "deal_category":
                    deal["deal_category"],

                "explanation":
                    explanation,

                "action":
                    action,
            }
        )

    recommendations = pd.DataFrame(
        rows
    )

    if recommendations.empty:

        raise ValueError(
            "No recommendations were generated."
        )

    recommendations = (
        recommendations
        .sort_values(
            by="recommendation_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return recommendations


# ============================================================
# SAVE
# ============================================================

def save_results(
    recommendations,
):

    recommendations.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "SMART RECOMMENDATION COMPLETE"
    )

    print()
    print("Output file:")
    print(OUTPUT_FILE)

    print()
    print(
        f"Recommendations generated: "
        f"{len(recommendations):,}"
    )


# ============================================================
# DISPLAY
# ============================================================

def display_results(
    recommendations,
):

    print()
    print("-" * 70)
    print("SMART SHOPPING INTELLIGENCE")
    print("-" * 70)

    print()
    print(
        "Recommendation distribution:"
    )

    distribution = (
        recommendations[
            "recommendation"
        ]
        .value_counts()
    )

    for recommendation, count in (
        distribution.items()
    ):

        print(
            f"  {recommendation}: {count}"
        )

    print()
    print(
        "Top recommendations:"
    )

    top = (
        recommendations
        .head(10)
    )

    for _, row in top.iterrows():

        print(
            f"{row['product_name']} | "
            f"{row['marketplace']} | "
            f"₹{row['current_price']:,.2f} | "
            f"Score: "
            f"{row['recommendation_score']:.2f} | "
            f"{row['recommendation']}"
        )

    print()
    print(
        "Best BUY NOW opportunities:"
    )

    buy_now = recommendations[
        recommendations[
            "recommendation"
        ]
        == "BUY NOW"
    ].head(5)

    if buy_now.empty:

        print(
            "  No BUY NOW recommendations."
        )

    else:

        for _, row in buy_now.iterrows():

            print(
                f"  {row['product_name']} | "
                f"{row['marketplace']} | "
                f"₹{row['current_price']:,.2f}"
            )

    print()
    print("-" * 70)
    print(
        "NEXT MODULE READY: ANALYTICS DATASET"
    )
    print("-" * 70)
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print_header()

    try:

        deals, prices = load_data()

        recommendations = (
            build_recommendations(
                deals,
                prices,
            )
        )

        save_results(
            recommendations
        )

        display_results(
            recommendations
        )

    except Exception as error:

        print()
        print("=" * 70)
        print(
            "SMART RECOMMENDATION FAILED"
        )
        print("=" * 70)
        print()
        print(str(error))
        print()

        raise


if __name__ == "__main__":
    main()