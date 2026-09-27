import json
import os

from google.cloud import bigquery
from anthropic import Anthropic
from dotenv import load_dotenv


load_dotenv()

PROJECT_ID = "novamart-customer-intelligence"


def get_segment_profiles():
    """Retrieve aggregated customer segment profiles from BigQuery."""

    client = bigquery.Client(project=PROJECT_ID)

    query = """
        SELECT
            s.segment_name,

            COUNT(*) AS customer_count,

            ROUND(AVG(m.total_orders), 2)
                AS avg_orders,

            ROUND(AVG(m.total_revenue), 2)
                AS avg_revenue,

            ROUND(AVG(m.average_order_value), 2)
                AS avg_order_value,

            ROUND(AVG(m.recency_days), 2)
                AS avg_recency_days,

            ROUND(AVG(m.category_count), 2)
                AS avg_category_count,

            ROUND(AVG(m.average_discount), 2)
                AS avg_discount,

            ROUND(AVG(m.return_rate) * 100, 2)
                AS avg_return_rate_pct,

            ROUND(
                COUNT(*) * 100.0 /
                SUM(COUNT(*)) OVER (),
                2
            ) AS customer_share_pct,

            ROUND(
                SUM(m.total_revenue),
                2
            ) AS total_revenue,

            ROUND(
                SUM(m.total_revenue) * 100.0 /
                SUM(SUM(m.total_revenue)) OVER (),
                2
            ) AS revenue_share_pct,

            ROUND(
                COUNTIF(m.recency_days <= 90) * 100.0 /
                COUNT(*),
                2
            ) AS purchased_within_90_days_pct,

            ROUND(
                COUNTIF(m.recency_days > 365) * 100.0 /
                COUNT(*),
                2
            ) AS inactive_365_plus_days_pct

        FROM
            `novamart-customer-intelligence.novamart_analytics.customer_metrics` AS m

        INNER JOIN
            `novamart-customer-intelligence.novamart_analytics.customer_segments` AS s
            ON m.customer_id = s.customer_id

        GROUP BY
            s.segment_name

        ORDER BY
            total_revenue DESC
    """

    results = client.query(query).result()

    segment_profiles = []

    for row in results:
        segment_profiles.append(dict(row))

    return segment_profiles

def generate_segment_insight(profile):
    """Generate structured AI business insights for a customer segment."""

    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY was not found. Check your .env file."
        )

    client = Anthropic(api_key=api_key)

    prompt = f"""
You are a customer analytics strategist for NovaMart, a fictional
e-commerce company.

Analyze the following customer segment using only the provided metrics.
Do not invent customer characteristics or unsupported facts.

SEGMENT DATA:
{json.dumps(profile, indent=2, default=str)}

Return a concise business analysis as valid JSON using exactly this structure:

{{
    "segment_name": "{profile['segment_name']}",
    "business_summary": "2-3 sentence summary of the segment",
    "key_characteristics": [
        "characteristic 1",
        "characteristic 2",
        "characteristic 3"
    ],
    "opportunity": "Primary business opportunity",
    "risk": "Primary business risk",
    "recommended_actions": [
        "action 1",
        "action 2",
        "action 3"
    ],
    "priority": "High, Medium, or Low"
}}

Requirements:
- Base every conclusion strictly on the supplied metrics.
- Treat total_revenue as observed historical revenue, NOT lifetime value or lifetime revenue.
- Treat all metrics as descriptive statistics, not evidence of causation.
- Do not infer customer motivations, satisfaction, price sensitivity, profitability,
  churn causes, or purchase intent unless directly supported by the supplied data.
- Do not invent demographics, product preferences, or behavioral explanations.
- Clearly distinguish observed facts from recommended future actions.
- Recommendations may propose future analyses or experiments, but do not claim
  their outcomes are known.
- Do not calculate or present hypothetical revenue values unless explicitly
  labeled as estimates derived from the supplied metrics.
- Focus recommendations on retention, reactivation, revenue, experimentation,
  and measurable marketing actions.
- Return exactly 3 key characteristics.
- Return exactly 3 recommended actions.
- Return only valid JSON with no Markdown formatting.
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    response_text = response.content[0].text.strip()

    # Remove Markdown code fences if Claude includes them
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[1]

        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()

    return json.loads(response_text)

def validate_insight(insight):
    """Validate the structure of an AI-generated segment insight."""

    required_fields = {
        "segment_name",
        "business_summary",
        "key_characteristics",
        "opportunity",
        "risk",
        "recommended_actions",
        "priority",
    }

    # Check required fields
    missing_fields = required_fields - insight.keys()

    if missing_fields:
        raise ValueError(
            f"AI response missing required fields: {missing_fields}"
        )

    # Check list lengths
    if len(insight["key_characteristics"]) != 3:
        raise ValueError(
            "AI response must contain exactly 3 key characteristics."
        )

    if len(insight["recommended_actions"]) != 3:
        raise ValueError(
            "AI response must contain exactly 3 recommended actions."
        )

    # Check priority value
    valid_priorities = {"High", "Medium", "Low"}

    if insight["priority"] not in valid_priorities:
        raise ValueError(
            f"Invalid priority: {insight['priority']}"
        )

    return True

if __name__ == "__main__":

    profiles = get_segment_profiles()

    print("Segment profiles retrieved from BigQuery!")
    print(f"Segments found: {len(profiles)}")

    insights = []

    for profile in profiles:

        print(
            f"\nGenerating AI insight for: "
            f"{profile['segment_name']}"
        )

        insight = generate_segment_insight(profile)

        validate_insight(insight)

        insights.append(insight)

        print("Complete!")

    output_file = "data/processed/segment_ai_insights.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            insights,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nAI insights saved to: {output_file}")

    print("\n" + "=" * 60)
    print("AI CUSTOMER SEGMENT INSIGHTS")
    print("=" * 60)

    print(json.dumps(insights, indent=2))