"""
SMARTPRICE AI
AI PRODUCT MATCHING ENGINE

Purpose:
    Match equivalent products across Amazon, Flipkart and Meesho.

Design:
    The raw dataset contains historical observations.
    Product matching should operate on unique product-marketplace
    identities rather than comparing every historical price record.

AI methods:
    1. TF-IDF character/word n-gram similarity
    2. Structured attribute similarity
    3. Model-number matching
    4. Weighted final match score

Output:
    outputs/product_matching_results.csv
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "marketplace_products_clean.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "product_matching_results.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

MARKETPLACES = [
    "Amazon",
    "Flipkart",
    "Meesho",
]

HIGH_MATCH_THRESHOLD = 0.85
POSSIBLE_MATCH_THRESHOLD = 0.65


ATTRIBUTE_WEIGHTS = {
    "brand": 0.15,
    "model_number": 0.30,
    "variant": 0.10,
    "storage": 0.15,
    "ram": 0.10,
    "color": 0.05,
    "category": 0.10,
    "subcategory": 0.05,
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Normalize product text for similarity comparison.
    """

    if pd.isna(value):
        return ""

    text = str(value).lower()

    # Replace separators with spaces.
    text = re.sub(r"[-_/|]+", " ", text)

    # Keep letters, numbers and spaces.
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# PRODUCT TEXT
# ============================================================

def build_product_text(row):
    """
    Build the text representation used by TF-IDF.
    """

    fields = [
        row.get("product_name", ""),
        row.get("brand", ""),
        row.get("model_number", ""),
        row.get("variant", ""),
        row.get("storage", ""),
        row.get("ram", ""),
        row.get("color", ""),
        row.get("category", ""),
        row.get("subcategory", ""),
    ]

    return " ".join(
        normalize_text(value)
        for value in fields
        if str(value).strip()
    )


# ============================================================
# ATTRIBUTE SIMILARITY
# ============================================================

def values_match(value_a, value_b):
    """
    Check whether two attribute values match.
    """

    a = normalize_text(value_a)
    b = normalize_text(value_b)

    if not a or not b:
        return 0.0

    return 1.0 if a == b else 0.0


def calculate_attribute_similarity(row_a, row_b):
    """
    Calculate weighted structured-attribute similarity.
    """

    score = 0.0
    total_weight = 0.0

    for attribute, weight in ATTRIBUTE_WEIGHTS.items():

        value_a = row_a.get(attribute, "")
        value_b = row_b.get(attribute, "")

        # If both attributes exist, include them.
        if (
            pd.notna(value_a)
            and pd.notna(value_b)
            and str(value_a).strip()
            and str(value_b).strip()
        ):
            match = values_match(
                value_a,
                value_b
            )

            score += weight * match
            total_weight += weight

    if total_weight == 0:
        return 0.0

    return score / total_weight


# ============================================================
# MODEL NUMBER
# ============================================================

def normalize_model_number(value):
    """
    Normalize model number for exact matching.
    """

    if pd.isna(value):
        return ""

    text = str(value).upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


def model_numbers_match(row_a, row_b):
    """
    Determine whether model numbers match exactly.
    """

    model_a = normalize_model_number(
        row_a.get("model_number", "")
    )

    model_b = normalize_model_number(
        row_b.get("model_number", "")
    )

    if not model_a or not model_b:
        return False

    return model_a == model_b


# ============================================================
# TEXT SIMILARITY
# ============================================================

def calculate_text_similarities(rows):
    """
    Calculate TF-IDF similarity for the unique
    product-marketplace rows.

    This is calculated once instead of repeatedly
    for thousands of historical rows.
    """

    texts = [
        build_product_text(row)
        for _, row in rows.iterrows()
    ]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(texts)

    similarity_matrix = cosine_similarity(
        matrix
    )

    return similarity_matrix


# ============================================================
# MATCH CATEGORY
# ============================================================

def classify_match(score):
    """
    Convert numerical match score into a business category.
    """

    if score >= HIGH_MATCH_THRESHOLD:
        return "High Match"

    if score >= POSSIBLE_MATCH_THRESHOLD:
        return "Possible Match"

    return "Low Match"


# ============================================================
# FINAL MATCH SCORE
# ============================================================

def calculate_match_score(
    text_score,
    attribute_score,
    exact_model_match,
):
    """
    Calculate final AI matching score.

    Text similarity:
        50%

    Structured attributes:
        50%

    Exact model-number match:
        Raises confidence to at least 0.90.
    """

    score = (
        0.50 * text_score
        + 0.50 * attribute_score
    )

    if exact_model_match:
        score = max(
            score,
            0.90
        )

    return round(
        float(score),
        4
    )


# ============================================================
# CREATE UNIQUE PRODUCT-MARKETPLACE DATA
# ============================================================

def prepare_unique_products(df):
    """
    Reduce historical observations to one identity
    record per product and marketplace.

    Product matching is about product identity,
    not daily price observations.
    """

    required_columns = [
        "product_id",
        "marketplace",
        "product_name",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # Keep the most recent observation for each
    # product-marketplace combination.
    if "scraped_at" in df.columns:

        df = df.sort_values(
            "scraped_at"
        )

        unique_df = (
            df
            .groupby(
                [
                    "product_id",
                    "marketplace",
                ],
                as_index=False
            )
            .tail(1)
            .copy()
        )

    else:

        unique_df = (
            df
            .drop_duplicates(
                subset=[
                    "product_id",
                    "marketplace",
                ]
            )
            .copy()
        )

    unique_df = unique_df.reset_index(
        drop=True
    )

    return unique_df


# ============================================================
# GENERATE CROSS-MARKETPLACE PAIRS
# ============================================================

def generate_product_pairs(unique_df):
    """
    Generate only cross-marketplace pairs.

    For each product identity:
        Amazon ↔ Flipkart
        Amazon ↔ Meesho
        Flipkart ↔ Meesho

    This produces 45 records at most for
    15 products × 3 marketplace combinations.
    """

    pairs = []

    grouped = unique_df.groupby(
        "product_id"
    )

    for product_id, group in grouped:

        group = group.copy()

        marketplace_rows = {}

        for marketplace in MARKETPLACES:

            marketplace_data = group[
                group["marketplace"]
                == marketplace
            ]

            if not marketplace_data.empty:

                marketplace_rows[
                    marketplace
                ] = marketplace_data.iloc[0]

        # ----------------------------------------------------
        # Amazon vs Flipkart
        # ----------------------------------------------------

        if (
            "Amazon" in marketplace_rows
            and "Flipkart" in marketplace_rows
        ):

            pairs.append(
                (
                    marketplace_rows["Amazon"],
                    marketplace_rows["Flipkart"],
                )
            )

        # ----------------------------------------------------
        # Amazon vs Meesho
        # ----------------------------------------------------

        if (
            "Amazon" in marketplace_rows
            and "Meesho" in marketplace_rows
        ):

            pairs.append(
                (
                    marketplace_rows["Amazon"],
                    marketplace_rows["Meesho"],
                )
            )

        # ----------------------------------------------------
        # Flipkart vs Meesho
        # ----------------------------------------------------

        if (
            "Flipkart" in marketplace_rows
            and "Meesho" in marketplace_rows
        ):

            pairs.append(
                (
                    marketplace_rows["Flipkart"],
                    marketplace_rows["Meesho"],
                )
            )

    return pairs


# ============================================================
# MATCH PRODUCTS
# ============================================================

def match_products(df):
    """
    Main product matching process.
    """

    print()
    print("Preparing unique product-marketplace records...")

    unique_df = prepare_unique_products(
        df
    )

    print(
        f"Unique product-marketplace records: "
        f"{len(unique_df):,}"
    )

    print()
    print(
        "Calculating TF-IDF similarity..."
    )

    similarity_matrix = (
        calculate_text_similarities(
            unique_df
        )
    )

    # Map dataframe index to matrix position.
    matrix_position = {
        index: position
        for position, index
        in enumerate(unique_df.index)
    }

    print(
        "TF-IDF similarity completed."
    )

    print()
    print(
        "Generating cross-marketplace pairs..."
    )

    pairs = generate_product_pairs(
        unique_df
    )

    print(
        f"Product pairs generated: "
        f"{len(pairs)}"
    )

    results = []

    # --------------------------------------------------------
    # Evaluate each pair
    # --------------------------------------------------------

    for row_a, row_b in pairs:

        index_a = row_a.name
        index_b = row_b.name

        position_a = matrix_position[
            index_a
        ]

        position_b = matrix_position[
            index_b
        ]

        text_score = similarity_matrix[
            position_a,
            position_b
        ]

        attribute_score = (
            calculate_attribute_similarity(
                row_a,
                row_b
            )
        )

        exact_model_match = (
            model_numbers_match(
                row_a,
                row_b
            )
        )

        match_score = calculate_match_score(
            text_score,
            attribute_score,
            exact_model_match,
        )

        match_category = classify_match(
            match_score
        )

        results.append(
            {
                "product_a_id":
                    row_a.get(
                        "product_id",
                        ""
                    ),

                "product_a_marketplace":
                    row_a.get(
                        "marketplace",
                        ""
                    ),

                "product_a_name":
                    row_a.get(
                        "product_name",
                        ""
                    ),

                "product_b_id":
                    row_b.get(
                        "product_id",
                        ""
                    ),

                "product_b_marketplace":
                    row_b.get(
                        "marketplace",
                        ""
                    ),

                "product_b_name":
                    row_b.get(
                        "product_name",
                        ""
                    ),

                "same_product_id":
                    (
                        row_a.get(
                            "product_id"
                        )
                        ==
                        row_b.get(
                            "product_id"
                        )
                    ),

                "text_similarity":
                    round(
                        float(
                            text_score
                        ),
                        4
                    ),

                "attribute_similarity":
                    round(
                        float(
                            attribute_score
                        ),
                        4
                    ),

                "model_number_match":
                    exact_model_match,

                "match_score":
                    match_score,

                "match_category":
                    match_category,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    return results_df


# ============================================================
# VALIDATE RESULTS
# ============================================================

def validate_results(results_df):
    """
    Validate the generated matching results.
    """

    if results_df.empty:
        raise ValueError(
            "No product matching results were generated."
        )

    required_columns = [
        "product_a_id",
        "product_a_marketplace",
        "product_a_name",
        "product_b_id",
        "product_b_marketplace",
        "product_b_name",
        "match_score",
        "match_category",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in results_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing output columns: "
            + ", ".join(missing_columns)
        )

    if results_df["match_score"].isna().any():
        raise ValueError(
            "Missing match scores detected."
        )

    if (
        results_df["match_score"]
        < 0
    ).any():

        raise ValueError(
            "Invalid negative match score detected."
        )

    if (
        results_df["match_score"]
        > 1
    ).any():

        raise ValueError(
            "Match score greater than 1 detected."
        )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_summary(results_df):
    """
    Display useful matching statistics.
    """

    print()
    print("=" * 70)
    print(
        "MATCHING RESULTS SUMMARY"
    )
    print("=" * 70)

    print()
    print(
        f"Total marketplace pairs: "
        f"{len(results_df)}"
    )

    print()
    print(
        "Match category distribution:"
    )

    print(
        results_df[
            "match_category"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Average match score:"
    )

    print(
        round(
            results_df[
                "match_score"
            ].mean(),
            4
        )
    )

    print()
    print(
        "High-confidence matches:"
    )

    high_matches = results_df[
        results_df[
            "match_category"
        ]
        == "High Match"
    ]

    print(
        len(high_matches)
    )

    print()
    print(
        "Top product matches:"
    )

    display_columns = [
        "product_a_marketplace",
        "product_a_name",
        "product_b_marketplace",
        "product_b_name",
        "match_score",
        "match_category",
    ]

    print(
        results_df
        .sort_values(
            "match_score",
            ascending=False
        )
        [display_columns]
        .head(15)
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "SMARTPRICE AI - PRODUCT MATCHING ENGINE"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    print()
    print(
        f"Input records: "
        f"{len(df):,}"
    )

    print(
        f"Unique products: "
        f"{df['product_id'].nunique()}"
    )

    print(
        "Marketplaces: "
        + ", ".join(
            sorted(
                df["marketplace"]
                .dropna()
                .unique()
            )
        )
    )

    # --------------------------------------------------------
    # Match products
    # --------------------------------------------------------

    results = match_products(
        df
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_results(
        results
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------

    display_summary(
        results
    )

    print()
    print("=" * 70)
    print(
        "PRODUCT MATCHING COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "Output file:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Next module:"
    )

    print(
        "Price Normalization"
    )


if __name__ == "__main__":
    main()