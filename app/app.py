"""
SMARTPRICE-AI
AI-Powered Multi-Marketplace Price Intelligence & Smart Shopping System
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartPrice-AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"

ANALYTICS_FILE = DATA_DIR / "analytics_dataset.csv"
HISTORY_FILE = DATA_DIR / "price_history.csv"
MATCHING_FILE = BASE_DIR / "outputs" / "product_matching_results.csv"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #1e3a8a 100%
        );
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: 2.6rem;
        margin-bottom: 0.4rem;
    }

    .hero p {
        font-size: 1.05rem;
        color: #dbeafe;
    }

    .metric-card {
        padding: 1.2rem;
        border-radius: 14px;
        background: white;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    .recommendation-buy {
        padding: 1rem;
        border-radius: 12px;
        background: #ecfdf5;
        border-left: 5px solid #10b981;
    }

    .recommendation-consider {
        padding: 1rem;
        border-radius: 12px;
        background: #fffbeb;
        border-left: 5px solid #f59e0b;
    }

    .recommendation-wait {
        padding: 1rem;
        border-radius: 12px;
        background: #fef2f2;
        border-left: 5px solid #ef4444;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_analytics():

    if not ANALYTICS_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(ANALYTICS_FILE)


@st.cache_data
def load_history():

    if not HISTORY_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(HISTORY_FILE)


@st.cache_data
def load_matching():

    if not MATCHING_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(MATCHING_FILE)


analytics_df = load_analytics()
history_df = load_history()
matching_df = load_matching()


# ============================================================
# VALIDATION
# ============================================================

if analytics_df.empty:

    st.error(
        "Analytics dataset not found. "
        "Run analytics_dataset.py before starting the application."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    # 🛒 SmartPrice-AI

    **AI-Powered Multi-Marketplace
    Price Intelligence**
    """
)

st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Executive Dashboard",
        "🔎 Product Search",
        "🏪 Marketplace Comparison",
        "📈 Price History",
        "🏷️ Deal Intelligence",
        "🤖 AI Recommendations",
        "💰 Savings Calculator",
        "📊 Analytics",
        "ℹ️ About",
    ],
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "SmartPrice-AI • Intelligent Shopping Analytics"
)


# ============================================================
# COMMON VALUES
# ============================================================

products = sorted(
    analytics_df["product_name"]
    .dropna()
    .unique()
)

marketplaces = sorted(
    analytics_df["marketplace"]
    .dropna()
    .unique()
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🛒 SmartPrice-AI</h1>
        <p>
        Compare marketplace prices, identify genuine deals,
        analyze price history, and make smarter purchasing decisions
        with AI-powered shopping intelligence.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if menu == "🏠 Executive Dashboard":

    st.markdown(
        '<div class="section-title">Executive Dashboard</div>',
        unsafe_allow_html=True,
    )

    total_products = analytics_df["product_id"].nunique()

    total_marketplaces = analytics_df[
        "marketplace"
    ].nunique()

    lowest_prices = analytics_df[
        analytics_df["is_lowest_price"] == True
    ]

    average_price = analytics_df[
        "effective_price"
    ].mean()

    average_savings = analytics_df[
        "marketplace_savings_percent"
    ].mean()

    buy_count = 0

    if "recommendation" in analytics_df.columns:

        buy_count = (
            analytics_df["recommendation"]
            .astype(str)
            .eq("BUY")
            .sum()
        )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Products",
            f"{total_products}"
        )

    with col2:
        st.metric(
            "Marketplaces",
            f"{total_marketplaces}"
        )

    with col3:
        st.metric(
            "Price Records",
            f"{len(analytics_df):,}"
        )

    with col4:
        st.metric(
            "Avg. Price",
            f"₹{average_price:,.0f}"
        )

    with col5:
        st.metric(
            "BUY Recommendations",
            f"{buy_count}"
        )

    st.markdown("### 🧠 Smart Shopping Overview")

    col1, col2 = st.columns(2)

    with col1:

        marketplace_counts = (
            lowest_prices[
                "marketplace"
            ]
            .value_counts()
            .reset_index()
        )

        marketplace_counts.columns = [
            "Marketplace",
            "Lowest Price Count",
        ]

        fig = px.bar(
            marketplace_counts,
            x="Marketplace",
            y="Lowest Price Count",
            title="Lowest-Price Marketplace Distribution",
            text="Lowest Price Count",
        )

        fig.update_layout(
            height=380,
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col2:

        recommendation_data = (
            analytics_df[
                "recommendation"
            ]
            .value_counts()
            .reset_index()
        )

        recommendation_data.columns = [
            "Recommendation",
            "Count",
        ]

        fig = px.pie(
            recommendation_data,
            names="Recommendation",
            values="Count",
            title="AI Recommendation Distribution",
            hole=0.45,
        )

        fig.update_layout(
            height=380,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.markdown("### 💡 Top Shopping Opportunities")

    display_columns = [
        "product_name",
        "marketplace",
        "effective_price",
        "deal_score",
        "recommendation_score",
        "recommendation",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in analytics_df.columns
    ]

    top_df = (
        analytics_df
        .sort_values(
            "recommendation_score",
            ascending=False,
        )
        .head(10)
        [display_columns]
        .copy()
    )

    if "effective_price" in top_df.columns:

        top_df["effective_price"] = (
            top_df["effective_price"]
            .map(lambda x: f"₹{x:,.2f}")
        )

    st.dataframe(
        top_df,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        f"Average potential marketplace savings: "
        f"{average_savings:.2f}%"
    )


# ============================================================
# PRODUCT SEARCH
# ============================================================

elif menu == "🔎 Product Search":

    st.markdown(
        '<div class="section-title">🔎 Product Search & AI Matching</div>',
        unsafe_allow_html=True,
    )

    selected_product = st.selectbox(
        "Search for a product",
        products,
    )

    product_data = analytics_df[
        analytics_df["product_name"]
        == selected_product
    ].copy()

    if product_data.empty:

        st.warning(
            "No marketplace records found."
        )

    else:

        st.markdown(
            f"### {selected_product}"
        )

        best_row = product_data.loc[
            product_data["effective_price"].idxmin()
        ]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Best Price",
                f"₹{best_row['effective_price']:,.2f}"
            )

        with col2:
            st.metric(
                "Best Marketplace",
                best_row["marketplace"]
            )

        with col3:
            st.metric(
                "Highest Price",
                f"₹{product_data['effective_price'].max():,.2f}"
            )

        with col4:
            savings = (
                product_data["effective_price"].max()
                - product_data["effective_price"].min()
            )

            st.metric(
                "Maximum Saving",
                f"₹{savings:,.2f}"
            )

        st.markdown("### Marketplace Listings")

        table = product_data[
            [
                "product_name",
                "marketplace",
                "effective_price",
                "rating",
                "is_lowest_price",
            ]
        ].copy()

        table = table.sort_values(
            "effective_price"
        )

        table["effective_price"] = (
            table["effective_price"]
            .map(lambda x: f"₹{x:,.2f}")
        )

        table["is_lowest_price"] = (
            table["is_lowest_price"]
            .map(
                lambda x: "🏆 Best Price"
                if x
                else ""
            )
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### 🤖 AI Product Matching")

        if not matching_df.empty:

            match_product = matching_df[
                matching_df.astype(str)
                .apply(
                    lambda row:
                    row.str.contains(
                        selected_product,
                        case=False,
                        na=False,
                    ).any(),
                    axis=1,
                )
            ]

            if not match_product.empty:

                st.dataframe(
                    match_product.head(10),
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.success(
                    "The product matching engine "
                    "has identified cross-marketplace "
                    "equivalent listings."
                )


# ============================================================
# MARKETPLACE COMPARISON
# ============================================================

elif menu == "🏪 Marketplace Comparison":

    st.markdown(
        '<div class="section-title">🏪 Marketplace Price Comparison</div>',
        unsafe_allow_html=True,
    )

    selected_product = st.selectbox(
        "Select product",
        products,
    )

    product_data = analytics_df[
        analytics_df["product_name"]
        == selected_product
    ].copy()

    if not product_data.empty:

        product_data = product_data.sort_values(
            "effective_price"
        )

        fig = px.bar(
            product_data,
            x="marketplace",
            y="effective_price",
            text="effective_price",
            title=f"Effective Price Comparison — {selected_product}",
        )

        fig.update_traces(
            texttemplate="₹%{text:,.0f}",
            textposition="outside",
        )

        fig.update_layout(
            yaxis_title="Effective Price (₹)",
            xaxis_title="Marketplace",
            height=450,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.markdown("### Comparison Details")

        comparison_columns = [
            "marketplace",
            "effective_price",
            "lowest_marketplace_price",
            "marketplace_price_rank",
            "savings_vs_highest_marketplace",
            "savings_vs_highest_percent",
        ]

        comparison_columns = [
            column
            for column in comparison_columns
            if column in product_data.columns
        ]

        comparison = product_data[
            comparison_columns
        ].copy()

        comparison["effective_price"] = (
            comparison["effective_price"]
            .map(lambda x: f"₹{x:,.2f}")
        )

        comparison["savings_vs_highest_marketplace"] = (
            comparison[
                "savings_vs_highest_marketplace"
            ]
            .map(lambda x: f"₹{x:,.2f}")
        )

        comparison["savings_vs_highest_percent"] = (
            comparison[
                "savings_vs_highest_percent"
            ]
            .map(lambda x: f"{x:.2f}%")
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PRICE HISTORY
# ============================================================

elif menu == "📈 Price History":

    st.markdown(
        '<div class="section-title">📈 Historical Price Intelligence</div>',
        unsafe_allow_html=True,
    )

    if history_df.empty:

        st.warning(
            "Price history data is unavailable."
        )

    else:

        # ----------------------------------------------------
        # Validate required historical columns
        # ----------------------------------------------------

        required_history_columns = [
            "product_name",
            "marketplace",
            "date",
            "normalized_price",
        ]

        missing_columns = [
            column
            for column in required_history_columns
            if column not in history_df.columns
        ]

        if missing_columns:

            st.error(
                "The price history dataset is missing "
                f"required columns: {missing_columns}"
            )

            st.stop()

        # ----------------------------------------------------
        # Prepare history data
        # ----------------------------------------------------

        history_display = history_df.copy()

        history_display["date"] = pd.to_datetime(
            history_display["date"],
            errors="coerce",
        )

        history_display["normalized_price"] = pd.to_numeric(
            history_display["normalized_price"],
            errors="coerce",
        )

        history_display = history_display.dropna(
            subset=[
                "date",
                "normalized_price",
            ]
        )

        # ----------------------------------------------------
        # Product selection
        # ----------------------------------------------------

        history_products = sorted(
            history_display[
                "product_name"
            ]
            .dropna()
            .unique()
        )

        selected_product = st.selectbox(
            "Select product",
            history_products,
        )

        product_history = history_display[
            history_display["product_name"]
            == selected_product
        ].copy()

        # ----------------------------------------------------
        # Marketplace selection
        # ----------------------------------------------------

        available_marketplaces = sorted(
            product_history[
                "marketplace"
            ]
            .dropna()
            .unique()
        )

        selected_marketplace = st.selectbox(
            "Select marketplace",
            available_marketplaces,
        )

        chart_data = product_history[
            product_history["marketplace"]
            == selected_marketplace
        ].copy()

        chart_data = chart_data.sort_values(
            "date"
        )

        # ----------------------------------------------------
        # Price history chart
        # ----------------------------------------------------

        if not chart_data.empty:

            fig = px.line(
                chart_data,
                x="date",
                y="normalized_price",
                markers=True,
                title=(
                    f"Price History — "
                    f"{selected_product} — "
                    f"{selected_marketplace}"
                ),
            )

            fig.update_layout(
                yaxis_title="Normalized Price (₹)",
                xaxis_title="Date",
                height=450,
                hovermode="x unified",
            )

            fig.update_traces(
                hovertemplate=(
                    "Date: %{x}<br>"
                    "Price: ₹%{y:,.2f}"
                    "<extra></extra>"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            # ------------------------------------------------
            # Historical metrics
            # ------------------------------------------------

            historical_low = chart_data[
                "normalized_price"
            ].min()

            historical_high = chart_data[
                "normalized_price"
            ].max()

            average_price = chart_data[
                "normalized_price"
            ].mean()

            current_price = chart_data[
                "normalized_price"
            ].iloc[-1]

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Current Price",
                    f"₹{current_price:,.2f}",
                )

            with col2:

                st.metric(
                    "Historical Low",
                    f"₹{historical_low:,.2f}",
                )

            with col3:

                st.metric(
                    "Historical High",
                    f"₹{historical_high:,.2f}",
                )

            with col4:

                st.metric(
                    "Average Price",
                    f"₹{average_price:,.2f}",
                )

            # ------------------------------------------------
            # Price position
            # ------------------------------------------------

            if historical_high > historical_low:

                price_position = (
                    (
                        current_price
                        - historical_low
                    )
                    /
                    (
                        historical_high
                        - historical_low
                    )
                ) * 100

            else:

                price_position = 50

            st.markdown(
                "### 📊 Current Price Position"
            )

            st.progress(
                min(
                    max(
                        int(price_position),
                        0,
                    ),
                    100,
                )
            )

            if current_price <= historical_low:

                st.success(
                    "🔥 The current price is at "
                    "the historical low."
                )

            elif current_price <= average_price:

                st.success(
                    "✅ The current price is "
                    "below the historical average."
                )

            elif current_price >= historical_high:

                st.warning(
                    "⚠️ The current price is at "
                    "the historical high."
                )

            else:

                st.info(
                    "ℹ️ The current price is within "
                    "the historical price range."
                )

            # ------------------------------------------------
            # Historical data table
            # ------------------------------------------------

            st.markdown(
                "### 📋 Historical Price Records"
            )

            table_columns = [
                "date",
                "normalized_price",
                "previous_price",
                "price_change",
                "price_change_percent",
                "trend",
            ]

            table_columns = [
                column
                for column in table_columns
                if column in chart_data.columns
            ]

            history_table = chart_data[
                table_columns
            ].sort_values(
                "date",
                ascending=False,
            ).copy()

            if "normalized_price" in history_table.columns:

                history_table[
                    "normalized_price"
                ] = history_table[
                    "normalized_price"
                ].map(
                    lambda x:
                    f"₹{x:,.2f}"
                )

            if "previous_price" in history_table.columns:

                history_table[
                    "previous_price"
                ] = pd.to_numeric(
                    history_table[
                        "previous_price"
                    ],
                    errors="coerce",
                )

                history_table[
                    "previous_price"
                ] = history_table[
                    "previous_price"
                ].map(
                    lambda x:
                    f"₹{x:,.2f}"
                    if pd.notna(x)
                    else "-"
                )

            if "price_change" in history_table.columns:

                history_table[
                    "price_change"
                ] = pd.to_numeric(
                    history_table[
                        "price_change"
                    ],
                    errors="coerce",
                )

                history_table[
                    "price_change"
                ] = history_table[
                    "price_change"
                ].map(
                    lambda x:
                    f"₹{x:,.2f}"
                    if pd.notna(x)
                    else "-"
                )

            if "price_change_percent" in history_table.columns:

                history_table[
                    "price_change_percent"
                ] = pd.to_numeric(
                    history_table[
                        "price_change_percent"
                    ],
                    errors="coerce",
                )

                history_table[
                    "price_change_percent"
                ] = history_table[
                    "price_change_percent"
                ].map(
                    lambda x:
                    f"{x:.2f}%"
                    if pd.notna(x)
                    else "-"
                )

            st.dataframe(
                history_table,
                use_container_width=True,
                hide_index=True,
            )

# ============================================================
# DEAL INTELLIGENCE
# ============================================================

elif menu == "🏷️ Deal Intelligence":

    st.markdown(
        '<div class="section-title">🏷️ AI Deal Intelligence</div>',
        unsafe_allow_html=True,
    )

    if "deal_score" in analytics_df.columns:

        deals = analytics_df.sort_values(
            "deal_score",
            ascending=False,
        ).copy()

        st.markdown(
            "### Top Detected Deals"
        )

        columns = [
            "product_name",
            "marketplace",
            "effective_price",
            "deal_score",
            "deal_category",
        ]

        columns = [
            column
            for column in columns
            if column in deals.columns
        ]

        display = deals[columns].head(15).copy()

        display["effective_price"] = (
            display["effective_price"]
            .map(lambda x: f"₹{x:,.2f}")
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            deals.head(15),
            x="deal_score",
            y="product_name",
            color="deal_category"
            if "deal_category" in deals.columns
            else None,
            orientation="h",
            title="Top Deal Scores",
        )

        fig.update_layout(
            height=600,
            yaxis_title="Product",
            xaxis_title="Deal Score",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# AI RECOMMENDATIONS
# ============================================================

elif menu == "🤖 AI Recommendations":

    st.markdown(
        '<div class="section-title">🤖 Smart Shopping Recommendations</div>',
        unsafe_allow_html=True,
    )

    recommendation_filter = st.selectbox(
        "Recommendation",
        [
            "All",
            "BUY",
            "CONSIDER",
            "WAIT",
        ],
    )

    recommendation_data = analytics_df.copy()

    if recommendation_filter != "All":

        recommendation_data = (
            recommendation_data[
                recommendation_data[
                    "recommendation"
                ]
                == recommendation_filter
            ]
        )

    recommendation_data = (
        recommendation_data
        .sort_values(
            "recommendation_score",
            ascending=False,
        )
    )

    columns = [
        "product_name",
        "marketplace",
        "effective_price",
        "recommendation_score",
        "recommendation",
        "recommendation_reason",
        "action",
    ]

    columns = [
        column
        for column in columns
        if column in recommendation_data.columns
    ]

    display = recommendation_data[
        columns
    ].copy()

    if "effective_price" in display.columns:

        display["effective_price"] = (
            display["effective_price"]
            .map(lambda x: f"₹{x:,.2f}")
        )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SAVINGS CALCULATOR
# ============================================================

elif menu == "💰 Savings Calculator":

    st.markdown(
        '<div class="section-title">💰 Smart Savings Calculator</div>',
        unsafe_allow_html=True,
    )

    selected_product = st.selectbox(
        "Select product",
        products,
    )

    product_data = analytics_df[
        analytics_df["product_name"]
        == selected_product
    ].copy()

    if not product_data.empty:

        lowest_price = product_data[
            "effective_price"
        ].min()

        highest_price = product_data[
            "effective_price"
        ].max()

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Lowest Available Price",
                f"₹{lowest_price:,.2f}",
            )

        with col2:

            st.metric(
                "Highest Available Price",
                f"₹{highest_price:,.2f}",
            )

        savings = highest_price - lowest_price

        savings_percent = (
            savings / highest_price * 100
            if highest_price > 0
            else 0
        )

        st.success(
            f"Potential saving by choosing the "
            f"lowest-priced marketplace: "
            f"₹{savings:,.2f} ({savings_percent:.2f}%)"
        )

        quantity = st.number_input(
            "Quantity",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
        )

        total_savings = savings * quantity

        st.metric(
            "Total Potential Saving",
            f"₹{total_savings:,.2f}",
        )


# ============================================================
# ANALYTICS
# ============================================================

elif menu == "📊 Analytics":

    st.markdown(
        '<div class="section-title">📊 Marketplace Analytics</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        marketplace_average = (
            analytics_df
            .groupby("marketplace")[
                "effective_price"
            ]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            marketplace_average,
            x="marketplace",
            y="effective_price",
            text="effective_price",
            title="Average Effective Price by Marketplace",
        )

        fig.update_traces(
            texttemplate="₹%{text:,.0f}",
            textposition="outside",
        )

        fig.update_layout(
            height=420,
            yaxis_title="Average Price (₹)",
            xaxis_title="Marketplace",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col2:

        lowest_distribution = (
            analytics_df[
                analytics_df["is_lowest_price"]
                == True
            ]
            ["marketplace"]
            .value_counts()
            .reset_index()
        )

        lowest_distribution.columns = [
            "Marketplace",
            "Count",
        ]

        fig = px.pie(
            lowest_distribution,
            names="Marketplace",
            values="Count",
            title="Products with Lowest Marketplace Price",
            hole=0.45,
        )

        fig.update_layout(
            height=420,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.markdown(
        "### Product Price Spread"
    )

    product_spread = (
        analytics_df
        .groupby("product_name")
        .agg(
            minimum_price=(
                "effective_price",
                "min"
            ),
            maximum_price=(
                "effective_price",
                "max"
            ),
        )
        .reset_index()
    )

    product_spread["price_difference"] = (
        product_spread["maximum_price"]
        - product_spread["minimum_price"]
    )

    product_spread = product_spread.sort_values(
        "price_difference",
        ascending=False,
    )

    fig = px.bar(
        product_spread,
        x="product_name",
        y="price_difference",
        title="Marketplace Price Difference by Product",
    )

    fig.update_layout(
        height=500,
        xaxis_title="Product",
        yaxis_title="Price Difference (₹)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# ABOUT
# ============================================================

elif menu == "ℹ️ About":

    st.markdown(
        '<div class="section-title">ℹ️ About SmartPrice-AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### SmartPrice-AI

        **AI-Powered Multi-Marketplace Price Intelligence &
        Smart Shopping System**

        SmartPrice-AI is a data science and machine learning
        application designed to help shoppers make better
        purchasing decisions.

        ### Core Intelligence

        - 🔎 AI-powered product matching
        - 🏪 Multi-marketplace price comparison
        - 💰 Effective customer-payable price calculation
        - 📈 Historical price analysis
        - 🏷️ Deal detection
        - 🤖 Smart purchase recommendations
        - 📊 Marketplace analytics
        - 💵 Savings analysis

        ### Technology Stack

        **Python • Pandas • NumPy • Scikit-learn • NLP •
        Plotly • Streamlit • PostgreSQL • Git/GitHub**

        ### Project Architecture

        The system follows a modular data-science pipeline:

        **Data Generation → Preprocessing → Product Matching →
        Price Normalization → Price History → Deal Detection →
        Smart Recommendation → Analytics → Dashboard**

        ### Important Data Note

        The current application uses development data generated
        for the project. It does not claim to retrieve live
        marketplace prices from Amazon, Flipkart, or Meesho.

        Real marketplace integrations would require appropriate
        permitted APIs or data sources and compliance with the
        relevant marketplace terms.
        """
    )