import json
import os
import re

from anthropic import Anthropic
from dotenv import load_dotenv
from google.cloud import bigquery


load_dotenv()


PROJECT_ID = "novamart-customer-intelligence"
DATASET_ID = "novamart_analytics"

ALLOWED_TABLES = {
    "novamart-customer-intelligence.novamart_analytics.customers",
    "novamart-customer-intelligence.novamart_analytics.products",
    "novamart-customer-intelligence.novamart_analytics.transactions",
    "novamart-customer-intelligence.novamart_analytics.customer_metrics",
    "novamart-customer-intelligence.novamart_analytics.customer_segments",
    "novamart-customer-intelligence.novamart_analytics.segment_ai_insights",
}

def generate_sql(question):
    """Convert a natural-language business question into BigQuery SQL."""

    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY was not found. Check your .env file."
        )

    client = Anthropic(api_key=api_key)

    prompt = f"""
You are a data analyst for NovaMart, a fictional e-commerce company.

Convert the user's business question into a Google BigQuery SQL query.

USER QUESTION:
{question}

You may query ONLY these tables:

1. `novamart-customer-intelligence.novamart_analytics.customers`

Columns:
- customer_id
- signup_date
- age
- gender
- city
- province
- acquisition_channel
- device_type

2. `novamart-customer-intelligence.novamart_analytics.products`

Columns:
- product_id
- product_name
- category
- subcategory
- price

3. `novamart-customer-intelligence.novamart_analytics.transactions`

Columns:
- transaction_id
- customer_id
- product_id
- transaction_date
- quantity
- unit_price
- discount_pct
- payment_method
- order_status

4. `novamart-customer-intelligence.novamart_analytics.customer_metrics`

Columns:
- customer_id
- signup_date
- age
- gender
- city
- province
- acquisition_channel
- device_type
- total_orders
- total_revenue
- average_order_value
- first_purchase_date
- last_purchase_date
- recency_days
- total_items
- average_discount
- category_count
- return_rate
- customer_tenure_days

5. `novamart-customer-intelligence.novamart_analytics.customer_segments`

Columns:
- customer_id
- cluster
- segment_name

6. `novamart-customer-intelligence.novamart_analytics.segment_ai_insights`

Columns:
- segment_name
- business_summary
- key_characteristics
- opportunity
- risk
- recommended_actions
- priority

Rules:
- Generate BigQuery Standard SQL.
- Generate exactly one SELECT query.
- SELECT statements and WITH clauses are allowed.
- Never generate INSERT, UPDATE, DELETE, MERGE, DROP, ALTER,
  CREATE, TRUNCATE, GRANT, REVOKE, or other data-changing statements.
- Use only the tables and columns listed above.
- Use fully qualified BigQuery table names.
- When calculating transaction revenue, use:
  quantity * unit_price * (1 - discount_pct / 100)
- Unless the question specifically concerns returns or cancellations,
  use only transactions where order_status = 'Completed'.
- Do not invent tables or columns.
- Do not answer the business question yourself.

Return valid JSON using exactly this structure:

{{
    "sql": "the generated BigQuery SQL query"
}}

Return only JSON with no Markdown formatting.
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    response_text = response.content[0].text.strip()

    # Remove Markdown fences if the model includes them.
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[1]

        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()

    result = json.loads(response_text)

    return result["sql"]

def validate_sql(sql):
    """Validate AI-generated SQL before allowing execution."""

    sql_upper = sql.upper().strip()

    # Query must begin with SELECT or WITH.
    if not (
        sql_upper.startswith("SELECT")
        or sql_upper.startswith("WITH")
    ):
        raise ValueError(
            "Unsafe SQL: query must begin with SELECT or WITH."
        )

    # Block statements that could modify data or database objects.
    forbidden_keywords = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "MERGE",
        "DROP",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "GRANT",
        "REVOKE",
    ]

    for keyword in forbidden_keywords:
        if keyword in sql_upper:
            raise ValueError(
                f"Unsafe SQL: forbidden keyword detected: {keyword}"
            )

    # Prevent multiple SQL statements.
    sql_without_final_semicolon = sql.strip().rstrip(";")

    if ";" in sql_without_final_semicolon:
        raise ValueError(
            "Unsafe SQL: multiple SQL statements detected."
        )

        # Extract fully qualified BigQuery table references.
    referenced_tables = set(
        re.findall(r"`([^`]+)`", sql)
    )

    if not referenced_tables:
        raise ValueError(
            "Unsafe SQL: no approved BigQuery table was referenced."
        )

    # Every referenced table must be explicitly approved.
    unauthorized_tables = referenced_tables - ALLOWED_TABLES

    if unauthorized_tables:
        raise ValueError(
            "Unsafe SQL: unauthorized table reference detected: "
            + ", ".join(sorted(unauthorized_tables))
        )
    
    print("SQL safety validation passed!")

    return True

def dry_run_query(sql):
    """Ask BigQuery to validate SQL without executing it."""

    client = bigquery.Client(project=PROJECT_ID)

    job_config = bigquery.QueryJobConfig(
        dry_run=True,
        use_query_cache=False,
    )

    try:
        query_job = client.query(
            sql,
            job_config=job_config,
        )

        print("BigQuery dry run passed!")
        print(
            f"Estimated bytes processed: "
            f"{query_job.total_bytes_processed:,}"
        )

        return True

    except Exception as error:
        raise ValueError(
            f"BigQuery dry run failed: {error}"
        )

def execute_query(sql):
    """Execute validated SQL and return the BigQuery results."""

    client = bigquery.Client(project=PROJECT_ID)

    print("Executing query...")

    query_job = client.query(sql)

    results = query_job.result()

    rows = [dict(row) for row in results]

    print(f"Query complete! Rows returned: {len(rows):,}")

    return rows

def generate_business_answer(question, results):
    """Convert BigQuery results into a concise business answer."""

    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY was not found. Check your .env file."
        )

    client = Anthropic(api_key=api_key)

    results_json = json.dumps(
        results,
        indent=2,
        default=str,
    )

    prompt = f"""
You are a data analyst for NovaMart.

Answer the user's business question using ONLY the BigQuery
results supplied below.

USER QUESTION:
{question}

BIGQUERY RESULTS:
{results_json}

Rules:
- Base every factual statement strictly on the supplied BigQuery results.
- Answer the user's question directly before providing supporting detail.
- Do not use knowledge encoded in segment names as evidence for conclusions.
- Do not introduce facts, metrics, causes, or explanations that are not
  explicitly present in the supplied results.
- You may perform simple arithmetic comparisons using values contained
  in the results.
- Do not infer customer motivations, purchase intent, satisfaction,
  profitability, price sensitivity, churn causes, or behavioral causes.
- Do not claim that one metric explains or causes another metric unless
  the supplied results directly establish that relationship.
- Do not provide recommendations, strategies, opportunities, or proposed
  actions unless the user explicitly asks for recommendations.
- Do not describe something as "best", "most valuable", "most important",
  or similar unless the user's requested metric directly defines that claim.
- Descriptive comparisons such as highest revenue, lowest AOV, or largest
  customer count are allowed when directly supported by the results.
- Format large dollar values in a business-friendly way.
- If multiple rows are returned, compare them when relevant.
- If the supplied results are insufficient to answer part of the question,
  explicitly say so rather than inferring the missing information.
- Keep the response concise and useful to a business stakeholder.
- Do not mention SQL unless necessary.

Return only the business answer.
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.content[0].text.strip()

if __name__ == "__main__":

    question = input(
        "Ask NovaMart a business question: "
    )

    sql = generate_sql(question)

    print("\nGenerated SQL:")
    print("=" * 60)
    print(sql)

    print("\nValidating SQL...")
    validate_sql(sql)

    print("\nRunning BigQuery dry run...")
    dry_run_query(sql)

    print("\nExecuting validated query...")
    results = execute_query(sql)

    print("\nQuery results:")
    print("=" * 60)

    for row in results:
        print(row)

    print("\nGenerating business answer...")

    answer = generate_business_answer(
        question,
        results,
    )

    print("\nAsk NovaMart:")
    print("=" * 60)
    print(answer)