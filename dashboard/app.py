import streamlit as st
from google.cloud import bigquery
import sys
from pathlib import Path
import altair as alt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.natural_language_analytics import (
    generate_sql,
    validate_sql,
    dry_run_query,
    execute_query,
    generate_business_answer,
)

from src.bigquery_client import get_bigquery_client

PROJECT_ID = "novamart-customer-intelligence"


@st.cache_data(ttl=600)
def get_executive_kpis():
    """Retrieve executive KPIs from BigQuery."""

    client = get_bigquery_client()

    query = """
    SELECT
        COUNT(DISTINCT customer_id) AS purchasing_customers,
        SUM(total_orders) AS completed_orders,
        SUM(total_revenue) AS total_revenue,
        SAFE_DIVIDE(
            SUM(total_revenue),
            SUM(total_orders)
        ) AS average_order_value
    FROM
        `novamart-customer-intelligence.novamart_analytics.customer_metrics`
    """

    result = client.query(query).result()

    return dict(next(iter(result)))

@st.cache_data(ttl=600)
def get_monthly_revenue():
    """Retrieve monthly completed-order revenue from BigQuery."""

    client = get_bigquery_client()

    query = """
    SELECT
        DATE_TRUNC(transaction_date, MONTH) AS month,
        SUM(
            quantity * unit_price * (1 - discount_pct / 100)
        ) AS revenue
    FROM
        `novamart-customer-intelligence.novamart_analytics.transactions`
    WHERE
        order_status = 'Completed'
    GROUP BY
        month
    ORDER BY
        month
    """

    result = client.query(query).result()

    return result.to_dataframe()

@st.cache_data(ttl=600)
def get_segment_summary():
    """Retrieve customer segment performance from BigQuery."""

    client = get_bigquery_client()

    query = """
    SELECT
        s.segment_name,
        COUNT(*) AS customer_count,
        SUM(m.total_revenue) AS total_revenue,
        AVG(m.total_revenue) AS avg_customer_revenue,
        AVG(m.average_order_value) AS avg_order_value,
        AVG(m.total_orders) AS avg_orders,
        AVG(m.recency_days) AS avg_recency_days
    FROM
        `novamart-customer-intelligence.novamart_analytics.customer_metrics` m
    JOIN
        `novamart-customer-intelligence.novamart_analytics.customer_segments` s
        ON m.customer_id = s.customer_id
    GROUP BY
        s.segment_name
    ORDER BY
        total_revenue DESC
    """

    result = client.query(query).result()

    return result.to_dataframe()

@st.cache_data(ttl=600)
def get_ai_segment_insights():
    """Retrieve Claude-generated segment insights from BigQuery."""

    client = get_bigquery_client()

    query = """
    SELECT
        segment_name,
        business_summary,
        key_characteristics,
        opportunity,
        risk,
        recommended_actions,
        priority
    FROM
        `novamart-customer-intelligence.novamart_analytics.segment_ai_insights`
    ORDER BY
        CASE priority
            WHEN 'High' THEN 1
            WHEN 'Medium' THEN 2
            WHEN 'Low' THEN 3
            ELSE 4
        END,
        segment_name
    """

    result = client.query(query).result()

    return result.to_dataframe()

@st.cache_data(ttl=600)
def get_category_revenue():
    """Retrieve completed-order revenue by product category."""

    client = get_bigquery_client()

    query = """
    SELECT
        p.category,
        SUM(
            t.quantity * t.unit_price * (1 - t.discount_pct / 100)
        ) AS revenue
    FROM
        `novamart-customer-intelligence.novamart_analytics.transactions` t
    JOIN
        `novamart-customer-intelligence.novamart_analytics.products` p
        ON t.product_id = p.product_id
    WHERE
        t.order_status = 'Completed'
    GROUP BY
        p.category
    ORDER BY
        revenue DESC
    """

    result = client.query(query).result()

    return result.to_dataframe()

@st.cache_data(ttl=600)
def get_channel_revenue():
    """Retrieve completed-order revenue by acquisition channel."""

    client = get_bigquery_client()

    query = """
    SELECT
        c.acquisition_channel,
        SUM(
            t.quantity * t.unit_price * (1 - t.discount_pct / 100)
        ) AS revenue
    FROM
        `novamart-customer-intelligence.novamart_analytics.transactions` t
    JOIN
        `novamart-customer-intelligence.novamart_analytics.customers` c
        ON t.customer_id = c.customer_id
    WHERE
        t.order_status = 'Completed'
    GROUP BY
        c.acquisition_channel
    ORDER BY
        revenue DESC
    """

    result = client.query(query).result()

    return result.to_dataframe()

st.set_page_config(
    page_title="NovaMart Customer Intelligence",
    page_icon="📊",
    layout="wide",
)

with st.sidebar:

    st.header("About NovaMart")

    st.write(
        "An end-to-end customer intelligence platform combining "
        "cloud analytics, machine learning, and generative AI."
    )

    st.divider()

    st.subheader("Technology Stack")

    st.markdown(
        """
        - **Google BigQuery** — cloud data warehouse
        - **Python** — analytics and application logic
        - **Scikit-learn** — customer segmentation
        - **Claude API** — AI insights and natural-language analytics
        - **Streamlit** — interactive dashboard
        - **Altair** — data visualization
        """
    )

    st.divider()

    st.subheader("Analytics Pipeline")

    st.markdown(
        """
        **Raw Data**  
        ↓  
        **BigQuery**  
        ↓  
        **Customer Analytics + K-Means**  
        ↓  
        **Claude AI**  
        ↓  
        **Business Intelligence Dashboard**
        """
    )

st.title("NovaMart Customer Intelligence")

st.write(
    "AI-powered customer analytics, segmentation, "
    "and natural-language business intelligence."
)

st.divider()

kpis = get_executive_kpis()

segment_kpis = get_segment_summary()

top_segment = segment_kpis.loc[
    segment_kpis["total_revenue"].idxmax()
]

top_segment_revenue_share = (
    top_segment["total_revenue"]
    / kpis["total_revenue"]
    * 100
)

st.subheader("Executive Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Revenue",
        f"${kpis['total_revenue'] / 1_000_000:.1f}M",
    )

with col2:
    st.metric(
        "Purchasing Customers",
        f"{kpis['purchasing_customers']:,}",
    )

with col3:
    st.metric(
        "Completed Orders",
        f"{kpis['completed_orders']:,}",
    )

with col4:
    st.metric(
        "Avg. Order Value",
        f"${kpis['average_order_value']:,.2f}",
    )

st.info(
    f"💡 **Executive Insight:** "
    f"{top_segment['segment_name']} customers generate "
    f"{top_segment_revenue_share:.1f}% of NovaMart's total revenue."
)

st.divider()

overview_tab, segments_tab, insights_tab, ask_tab = st.tabs(
    [
        "📈 Overview",
        "👥 Customer Segments",
        "🤖 AI Insights",
        "💬 Ask NovaMart",
    ]
)

with overview_tab:

    st.header("Business Performance")

    st.caption(
        "Monitor NovaMart's revenue trends and identify the product categories "
        "and acquisition channels contributing the most revenue."
    )

    # Monthly Revenue Chart
    monthly_revenue = get_monthly_revenue()

    st.subheader("Monthly Revenue")

    monthly_revenue_chart = (
        alt.Chart(monthly_revenue)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "month:T",
                title="Month",
            ),
            y=alt.Y(
                "revenue:Q",
                title="Revenue ($)",
            ),
            tooltip=[
                alt.Tooltip(
                    "month:T",
                    title="Month",
                    format="%B %Y",
                ),
                alt.Tooltip(
                    "revenue:Q",
                    title="Revenue",
                    format="$,.2f",
                ),
            ],
        )
    )

    st.altair_chart(
        monthly_revenue_chart,
        width="stretch",
    )

    # Product Category and Acquisition Channel Charts
    col1, col2 = st.columns(2)

    # Revenue by Product Category Chart
    with col1:
        category_revenue = get_category_revenue()

        category_revenue = category_revenue.sort_values(
            "revenue",
            ascending=False,
        )

        st.subheader("Revenue by Product Category")

        category_chart = (
            alt.Chart(category_revenue)
            .mark_bar()
            .encode(
                x=alt.X(
                    "category:N",
                    sort="-y",
                    title="Product Category",
                ),
                y=alt.Y(
                    "revenue:Q",
                    title="Revenue ($)",
                ),
                tooltip=[
                    alt.Tooltip(
                        "category:N",
                        title="Category",
                    ),
                    alt.Tooltip(
                        "revenue:Q",
                        title="Revenue",
                        format="$,.2f",
                    ),
                ],
            )
        )

        st.altair_chart(
            category_chart,
            width="stretch",
        )

    # Revenue by Acquisition Channel Chart
    with col2:
        channel_revenue = get_channel_revenue()

        channel_revenue = channel_revenue.sort_values(
            "revenue",
            ascending=False,
        )

        st.subheader("Revenue by Acquisition Channel")

        channel_chart = (
            alt.Chart(channel_revenue)
            .mark_bar()
            .encode(
                x=alt.X(
                    "acquisition_channel:N",
                    sort="-y",
                    title="Acquisition Channel",
                ),
                y=alt.Y(
                    "revenue:Q",
                    title="Revenue ($)",
                ),
                tooltip=[
                    alt.Tooltip(
                        "acquisition_channel:N",
                        title="Channel",
                    ),
                    alt.Tooltip(
                        "revenue:Q",
                        title="Revenue",
                        format="$,.2f",
                    ),
                ],
            )
        )

        st.altair_chart(
            channel_chart,
            width="stretch",
        )

with segments_tab:

    st.header("Customer Segmentation")

    st.caption(
        "Explore behavioral customer segments identified using K-Means clustering "
        "across purchasing, revenue, recency, category, and discount behavior."
    )

    segment_summary = get_segment_summary()

    st.subheader("Customer Segments")

    st.dataframe(
        segment_summary,
        width="stretch",
        hide_index=True,
        column_config={
            "segment_name": st.column_config.TextColumn(
                "Customer Segment"
            ),
            "customer_count": st.column_config.NumberColumn(
                "Customers",
                format="%d",
            ),
            "total_revenue": st.column_config.NumberColumn(
                "Total Revenue",
                format="$%.2f",
            ),
            "avg_customer_revenue": st.column_config.NumberColumn(
                "Avg. Customer Revenue",
                format="$%.2f",
            ),
            "avg_order_value": st.column_config.NumberColumn(
                "Avg. Order Value",
                format="$%.2f",
            ),
            "avg_orders": st.column_config.NumberColumn(
                "Avg. Orders",
                format="%.2f",
            ),
            "avg_recency_days": st.column_config.NumberColumn(
                "Avg. Recency (Days)",
                format="%.1f",
            ),
        },
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue by Customer Segment")

        segment_revenue_chart = (
            alt.Chart(segment_summary)
            .mark_bar()
            .encode(
                x=alt.X(
                    "segment_name:N",
                    sort="-y",
                    title="Customer Segment",
                ),
                y=alt.Y(
                    "total_revenue:Q",
                    title="Revenue ($)",
                ),
                tooltip=[
                    alt.Tooltip(
                        "segment_name:N",
                        title="Segment",
                    ),
                    alt.Tooltip(
                        "total_revenue:Q",
                        title="Revenue",
                        format="$,.2f",
                    ),
                ],
            )
        )

        st.altair_chart(
            segment_revenue_chart,
            width="stretch",
        )

    with col2:
        st.subheader("Customers by Segment")

        segment_customers_chart = (
            alt.Chart(segment_summary)
            .mark_bar()
            .encode(
                x=alt.X(
                    "segment_name:N",
                    sort="-y",
                    title="Customer Segment",
                ),
                y=alt.Y(
                    "customer_count:Q",
                    title="Number of Customers",
                ),
                tooltip=[
                    alt.Tooltip(
                        "segment_name:N",
                        title="Segment",
                    ),
                    alt.Tooltip(
                        "customer_count:Q",
                        title="Customers",
                        format=",",
                    ),
                ],
            )
        )

        st.altair_chart(
            segment_customers_chart,
            width="stretch",
        )

with insights_tab:

    st.header("AI-Powered Segment Insights")

    st.caption(
        "Claude interprets aggregated customer segment metrics to identify "
        "business opportunities, risks, and recommended actions. "
        "Customer-level data is not sent to the AI model."
    )

    ai_insights = get_ai_segment_insights()

    for _, insight in ai_insights.iterrows():

        with st.container(border=True):

            col1, col2 = st.columns([4, 1])

            with col1:
                st.subheader(insight["segment_name"])

            with col2:
                priority = insight["priority"]

                if priority == "High":
                    st.error("🔴 High Priority")
                elif priority == "Medium":
                    st.warning("🟠 Medium Priority")
                else:
                    st.success("🟢 Low Priority")

            st.write(insight["business_summary"])

            st.markdown("**Key Characteristics**")

            for characteristic in insight["key_characteristics"]:
                st.write(f"• {characteristic}")

            st.markdown("**Opportunity**")
            st.write(insight["opportunity"])

            st.markdown("**Risk**")
            st.write(insight["risk"])

            st.markdown("**Recommended Actions**")

            for action in insight["recommended_actions"]:
                st.write(f"• {action}")

with ask_tab:

    st.header("Ask NovaMart")

    st.caption(
        "Ask a business question in plain English. Claude translates the question "
        "into validated SQL, BigQuery retrieves the underlying data, and Claude "
        "uses those query results to generate a grounded business answer."
    )

    st.caption(
        "*Try asking: "
        "\"Which customer segment generated the most revenue?\" · "
        "\"Which product category generated the most revenue?\" · "
        "\"Compare average order value across customer segments.\"*"
    )

    question = st.text_input(
        "Business Question",
        placeholder="e.g. Which customer segment generated the most revenue?",
    )

    ask_button = st.button(
        "Ask NovaMart",
        type="primary",
    )

    if ask_button:

        if not question.strip():
            st.warning("Please enter a business question.")

        else:
            try:
                with st.spinner("Analyzing NovaMart data..."):

                    # 1. Generate SQL from the user's question
                    sql = generate_sql(question)

                    # 2. Validate the generated SQL
                    validate_sql(sql)

                    # 3. Test the query before execution
                    dry_run_query(sql)

                    # 4. Execute the query in BigQuery
                    results = execute_query(sql)

                    # 5. Generate a grounded business answer
                    answer = generate_business_answer(
                        question,
                        results,
                    )

                st.subheader("Answer")

                safe_answer = answer.replace("$", r"\$")

                st.markdown(safe_answer)

                with st.expander("View Query Results"):
                    st.dataframe(
                        results,
                        width="stretch",
                        hide_index=True,
                    )

                with st.expander("View Generated SQL"):
                    st.code(sql, language=None)

            except Exception as error:
                st.error(
                    f"Unable to answer the question: {error}"
                )