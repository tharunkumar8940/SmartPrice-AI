from pathlib import Path
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RAW_DIR / "marketplace_products.csv"


# ============================================================
# PRODUCT CATALOG
# ============================================================

PRODUCTS = [
    {
        "product_id": "SP001",
        "brand": "Samsung",
        "category": "Smartphone",
        "subcategory": "Android Phone",
        "product_name": "Samsung Galaxy S24 5G",
        "model_number": "SM-S921B",
        "variant": "256GB",
        "color": "Black",
        "storage": "256GB",
        "ram": "8GB",
        "base_price": 74999,
    },
    {
        "product_id": "SP002",
        "brand": "Apple",
        "category": "Smartphone",
        "subcategory": "iPhone",
        "product_name": "Apple iPhone 15",
        "model_number": "A3090",
        "variant": "128GB",
        "color": "Black",
        "storage": "128GB",
        "ram": "6GB",
        "base_price": 69999,
    },
    {
        "product_id": "SP003",
        "brand": "OnePlus",
        "category": "Smartphone",
        "subcategory": "Android Phone",
        "product_name": "OnePlus 13",
        "model_number": "CPH2653",
        "variant": "256GB",
        "color": "Black",
        "storage": "256GB",
        "ram": "12GB",
        "base_price": 69999,
    },
    {
        "product_id": "SP004",
        "brand": "Google",
        "category": "Smartphone",
        "subcategory": "Android Phone",
        "product_name": "Google Pixel 9",
        "model_number": "GRY0E",
        "variant": "256GB",
        "color": "Obsidian",
        "storage": "256GB",
        "ram": "12GB",
        "base_price": 79999,
    },
    {
        "product_id": "SP005",
        "brand": "Sony",
        "category": "Headphones",
        "subcategory": "Wireless Headphones",
        "product_name": "Sony WH-1000XM5",
        "model_number": "WH1000XM5",
        "variant": "Standard",
        "color": "Black",
        "storage": "N/A",
        "ram": "N/A",
        "base_price": 29990,
    },
    {
        "product_id": "SP006",
        "brand": "Apple",
        "category": "Laptop",
        "subcategory": "MacBook",
        "product_name": "Apple MacBook Air M3",
        "model_number": "MRXV3HN/A",
        "variant": "256GB",
        "color": "Space Grey",
        "storage": "256GB",
        "ram": "8GB",
        "base_price": 114900,
    },
    {
        "product_id": "SP007",
        "brand": "HP",
        "category": "Laptop",
        "subcategory": "Windows Laptop",
        "product_name": "HP Pavilion 14",
        "model_number": "14-ec1003AU",
        "variant": "512GB",
        "color": "Silver",
        "storage": "512GB",
        "ram": "16GB",
        "base_price": 67990,
    },
    {
        "product_id": "SP008",
        "brand": "Dell",
        "category": "Laptop",
        "subcategory": "Windows Laptop",
        "product_name": "Dell Inspiron 15",
        "model_number": "3520",
        "variant": "512GB",
        "color": "Silver",
        "storage": "512GB",
        "ram": "16GB",
        "base_price": 62990,
    },
    {
        "product_id": "SP009",
        "brand": "boAt",
        "category": "Audio",
        "subcategory": "Wireless Earbuds",
        "product_name": "boAt Airdopes 141",
        "model_number": "Airdopes141",
        "variant": "Standard",
        "color": "Black",
        "storage": "N/A",
        "ram": "N/A",
        "base_price": 4499,
    },
    {
        "product_id": "SP010",
        "brand": "JBL",
        "category": "Audio",
        "subcategory": "Wireless Earbuds",
        "product_name": "JBL Tune 230NC",
        "model_number": "T230NC",
        "variant": "Standard",
        "color": "Black",
        "storage": "N/A",
        "ram": "N/A",
        "base_price": 5999,
    },
    {
        "product_id": "SP011",
        "brand": "Nike",
        "category": "Fashion",
        "subcategory": "Running Shoes",
        "product_name": "Nike Revolution 7",
        "model_number": "FB2207",
        "variant": "Standard",
        "color": "Black",
        "storage": "N/A",
        "ram": "N/A",
        "base_price": 4995,
    },
    {
        "product_id": "SP012",
        "brand": "Adidas",
        "category": "Fashion",
        "subcategory": "Running Shoes",
        "product_name": "Adidas Duramo SL",
        "model_number": "IE9690",
        "variant": "Standard",
        "color": "Black",
        "storage": "N/A",
        "ram": "N/A",
        "base_price": 5999,
    },
    {
        "product_id": "SP013",
        "brand": "Xiaomi",
        "category": "Smart TV",
        "subcategory": "4K Television",
        "product_name": "Xiaomi 55 Inch 4K Smart TV",
        "model_number": "X55",
        "variant": "55 Inch",
        "color": "Black",
        "storage": "32GB",
        "ram": "2GB",
        "base_price": 44999,
    },
    {
        "product_id": "SP014",
        "brand": "LG",
        "category": "Smart TV",
        "subcategory": "4K Television",
        "product_name": "LG 55 Inch 4K Smart TV",
        "model_number": "55UR7500",
        "variant": "55 Inch",
        "color": "Black",
        "storage": "N/A",
        "ram": "N/A",
        "base_price": 54999,
    },
    {
        "product_id": "SP015",
        "brand": "Philips",
        "category": "Home Appliance",
        "subcategory": "Air Fryer",
        "product_name": "Philips Air Fryer",
        "model_number": "HD9200",
        "variant": "4.1L",
        "color": "Black",
        "storage": "N/A",
        "ram": "N/A",
        "base_price": 9999,
    },
]


# ============================================================
# MARKETPLACE SETTINGS
# ============================================================

MARKETPLACES = {
    "Amazon": {
        "price_multiplier": 1.00,
        "shipping_range": (0, 99),
        "coupon_range": (0, 1500),
    },
    "Flipkart": {
        "price_multiplier": 0.98,
        "shipping_range": (0, 79),
        "coupon_range": (0, 1800),
    },
    "Meesho": {
        "price_multiplier": 0.96,
        "shipping_range": (0, 69),
        "coupon_range": (0, 1200),
    },
}


# ============================================================
# MARKETPLACE PRODUCT NAME VARIATIONS
# ============================================================

def create_product_name(product, marketplace):
    """Create realistic marketplace-specific product names."""

    name = product["product_name"]

    if marketplace == "Amazon":
        return (
            f"{name} {product['variant']} "
            f"{product['color']}"
        )

    if marketplace == "Flipkart":
        replacements = {
            "Samsung Galaxy S24 5G": "Samsung S24 5G Smartphone",
            "Apple iPhone 15": "Apple iPhone 15 Smartphone",
            "OnePlus 13": "OnePlus 13 5G Smartphone",
            "Google Pixel 9": "Google Pixel 9 5G Phone",
            "Sony WH-1000XM5": "Sony WH1000XM5 Wireless Headphones",
            "Apple MacBook Air M3": "Apple MacBook Air M3 Laptop",
            "HP Pavilion 14": "HP Pavilion 14 Laptop",
            "Dell Inspiron 15": "Dell Inspiron 15 Laptop",
            "boAt Airdopes 141": "boAt Airdopes 141 Bluetooth Earbuds",
            "JBL Tune 230NC": "JBL Tune 230NC TWS Earbuds",
            "Nike Revolution 7": "Nike Revolution 7 Running Shoes",
            "Adidas Duramo SL": "Adidas Duramo SL Running Shoes",
            "Xiaomi 55 Inch 4K Smart TV": "Xiaomi 55 Inch 4K Ultra HD Smart TV",
            "LG 55 Inch 4K Smart TV": "LG 55 Inch 4K UHD Smart Television",
            "Philips Air Fryer": "Philips HD9200 Air Fryer",
        }

        return replacements.get(name, name)

    if marketplace == "Meesho":
        replacements = {
            "Samsung Galaxy S24 5G": "Samsung Galaxy S24 256GB Mobile",
            "Apple iPhone 15": "Apple iPhone 15 128GB Mobile",
            "OnePlus 13": "OnePlus 13 256GB Smartphone",
            "Google Pixel 9": "Google Pixel 9 256GB Smartphone",
            "Sony WH-1000XM5": "Sony Wireless Noise Cancelling Headphones",
            "Apple MacBook Air M3": "Apple MacBook Air M3 8GB Laptop",
            "HP Pavilion 14": "HP Pavilion 14 16GB Laptop",
            "Dell Inspiron 15": "Dell Inspiron 15 16GB Laptop",
            "boAt Airdopes 141": "boAt Airdopes 141 Wireless Earbuds",
            "JBL Tune 230NC": "JBL Tune 230NC Bluetooth Earbuds",
            "Nike Revolution 7": "Nike Revolution 7 Sports Running Shoes",
            "Adidas Duramo SL": "Adidas Duramo SL Sports Shoes",
            "Xiaomi 55 Inch 4K Smart TV": "Xiaomi 55 Inch 4K Smart LED TV",
            "LG 55 Inch 4K Smart TV": "LG 55 Inch 4K Smart LED Television",
            "Philips Air Fryer": "Philips HD9200 4.1L Air Fryer",
        }

        return replacements.get(name, name)

    return name


# ============================================================
# PRICE GENERATION
# ============================================================

def generate_price(product, marketplace, day_index):
    """Generate a realistic marketplace price."""

    settings = MARKETPLACES[marketplace]

    seasonal_factor = 1 + (
        0.015 * np.sin(day_index / 5)
    )

    random_factor = np.random.uniform(0.96, 1.04)

    base_price = (
        product["base_price"]
        * settings["price_multiplier"]
        * seasonal_factor
        * random_factor
    )

    original_price = round(base_price * np.random.uniform(1.08, 1.25))

    discount_percent = round(
        ((original_price - base_price) / original_price) * 100,
        2,
    )

    shipping_cost = round(
        np.random.uniform(
            settings["shipping_range"][0],
            settings["shipping_range"][1],
        )
    )

    coupon_discount = round(
        np.random.uniform(
            settings["coupon_range"][0],
            settings["coupon_range"][1],
        )
    )

    # Keep coupons realistic and proportional to the selling price.
    coupon_cap = np.floor(base_price * 0.05 * 100) / 100
    coupon_discount = min(
        coupon_discount,
        coupon_cap,
    )

    # Calculate the actual customer-payable price.
    effective_price = (
        base_price
        + shipping_cost
        - coupon_discount
    )

    # Never allow discounts and coupons to reduce the price
    # below 85% of the generated selling price.
    minimum_effective_price = base_price * 0.85

    effective_price = max(
        effective_price,
        minimum_effective_price,
    )

    return {
        "original_price": round(original_price, 2),
        "discount_percent": discount_percent,
        "selling_price": round(base_price, 2),
        "shipping_cost": round(shipping_cost, 2),
        "coupon_discount": round(coupon_discount, 2),
        "effective_price": round(effective_price, 2),
    }


# ============================================================
# DATASET CREATION
# ============================================================

def create_dataset(days=90):
    """Create historical marketplace product data."""

    rows = []

    start_date = datetime.now() - timedelta(days=days - 1)

    for day_index in range(days):

        current_date = start_date + timedelta(days=day_index)

        for product in PRODUCTS:

            for marketplace in MARKETPLACES:

                price_data = generate_price(
                    product,
                    marketplace,
                    day_index,
                )

                availability = np.random.choice(
                    ["In Stock", "Limited Stock", "Out of Stock"],
                    p=[0.90, 0.08, 0.02],
                )

                if availability == "Out of Stock":
                    price_data["selling_price"] = np.nan
                    price_data["effective_price"] = np.nan

                product_url = (
                    f"https://example.com/"
                    f"{marketplace.lower()}/"
                    f"{product['product_id'].lower()}"
                )

                rows.append(
                    {
                        "product_id": product["product_id"],
                        "marketplace": marketplace,
                        "product_name": create_product_name(
                            product,
                            marketplace,
                        ),
                        "brand": product["brand"],
                        "category": product["category"],
                        "subcategory": product["subcategory"],
                        "model_number": product["model_number"],
                        "variant": product["variant"],
                        "color": product["color"],
                        "storage": product["storage"],
                        "ram": product["ram"],
                        "rating": round(
                            np.random.uniform(3.5, 4.9),
                            1,
                        ),
                        "review_count": int(
                            np.random.randint(50, 250000)
                        ),
                        "original_price": price_data[
                            "original_price"
                        ],
                        "discount_percent": price_data[
                            "discount_percent"
                        ],
                        "selling_price": price_data[
                            "selling_price"
                        ],
                        "shipping_cost": price_data[
                            "shipping_cost"
                        ],
                        "coupon_discount": price_data[
                            "coupon_discount"
                        ],
                        "effective_price": price_data[
                            "effective_price"
                        ],
                        "availability": availability,
                        "product_url": product_url,
                        "scraped_at": current_date.strftime(
                            "%Y-%m-%d"
                        ),
                    }
                )

    df = pd.DataFrame(rows)

    # Ensure correct data types.
    df["scraped_at"] = pd.to_datetime(df["scraped_at"])

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

    # Sort the final dataset.
    df = df.sort_values(
        [
            "product_id",
            "scraped_at",
            "marketplace",
        ]
    ).reset_index(drop=True)

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(df):
    """Validate the generated dataset."""

    required_columns = [
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

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["product_id"].isna().any():
        raise ValueError(
            "product_id contains missing values."
        )

    if df["marketplace"].isna().any():
        raise ValueError(
            "marketplace contains missing values."
        )

    valid_marketplaces = set(MARKETPLACES.keys())

    detected_marketplaces = set(
        df["marketplace"].dropna().unique()
    )

    if not detected_marketplaces.issubset(
        valid_marketplaces
    ):
        raise ValueError(
            "Unexpected marketplace detected."
        )

    duplicate_columns = [
        "product_id",
        "marketplace",
        "scraped_at",
    ]

    duplicate_count = df.duplicated(
        subset=duplicate_columns
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate product records."
        )

    if (
        df["effective_price"].dropna() < 0
    ).any():
        raise ValueError(
            "Negative effective prices detected."
        )

    # Effective price must remain at least 85% of selling price.
    valid_price_rows = (
        df["selling_price"].notna()
        & df["effective_price"].notna()
    )

    if (
        df.loc[valid_price_rows, "effective_price"]
        < df.loc[valid_price_rows, "selling_price"] * 0.85
    ).any():
        raise ValueError(
            "Unrealistic effective prices detected."
        )

    # Coupon must never exceed 5% of selling price.
    valid_coupon_rows = (
        df["selling_price"].notna()
        & df["coupon_discount"].notna()
    )

    coupon_limit = (
        df.loc[valid_coupon_rows, "selling_price"] * 0.05
    )

    if (
        df.loc[valid_coupon_rows, "coupon_discount"]
        > coupon_limit + 0.01
    ).any():
        raise ValueError(
            "Unrealistic coupon discounts detected."
        )

    if (
        df["rating"].dropna() < 0
    ).any() or (
        df["rating"].dropna() > 5
    ).any():
        raise ValueError(
            "Invalid rating detected."
        )

    return True


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("SMARTPRICE AI - DATASET GENERATOR")
    print("=" * 60)

    df = create_dataset(days=90)

    validate_dataset(df)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nDataset created successfully.")
    print(f"Location: {OUTPUT_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nProducts:")
    print(df["product_id"].nunique())

    print("\nMarketplaces:")
    print(
        ", ".join(
            sorted(df["marketplace"].unique())
        )
    )

    print("\nDate range:")
    print(
        f"{df['scraped_at'].min().date()} "
        f"to "
        f"{df['scraped_at'].max().date()}"
    )

    print("\nDataset preview:")
    print(
        df[
            [
                "product_id",
                "marketplace",
                "product_name",
                "selling_price",
                "effective_price",
                "rating",
            ]
        ].head(10).to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("DATASET READY")
    print("=" * 60)


if __name__ == "__main__":
    main()