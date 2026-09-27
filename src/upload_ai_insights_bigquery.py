import json

import pandas as pd
from google.cloud import bigquery


PROJECT_ID = "novamart-customer-intelligence"
DATASET_ID = "novamart_analytics"
TABLE_ID = "segment_ai_insights"

INSIGHTS_FILE = "data/processed/segment_ai_insights.json"


def prepare_ai_insights():
    """Prepare validated AI segment insights for BigQuery."""

    with open(INSIGHTS_FILE, "r", encoding="utf-8") as file:
        insights = json.load(file)

    rows = []

    for insight in insights:
        rows.append(
            {
                "segment_name": insight["segment_name"],
                "business_summary": insight["business_summary"],
                "key_characteristics": insight["key_characteristics"],
                "opportunity": insight["opportunity"],
                "risk": insight["risk"],
                "recommended_actions": insight["recommended_actions"],
                "priority": insight["priority"],
            }
        )

    return pd.DataFrame(rows)


def upload_to_bigquery(insights_df):
    """Upload AI-generated segment insights to BigQuery."""

    client = bigquery.Client(project=PROJECT_ID)

    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

    schema = [
        bigquery.SchemaField(
            "segment_name",
            "STRING",
            mode="REQUIRED",
        ),
        bigquery.SchemaField(
            "business_summary",
            "STRING",
            mode="REQUIRED",
        ),
        bigquery.SchemaField(
            "key_characteristics",
            "STRING",
            mode="REPEATED",
        ),
        bigquery.SchemaField(
            "opportunity",
            "STRING",
            mode="REQUIRED",
        ),
        bigquery.SchemaField(
            "risk",
            "STRING",
            mode="REQUIRED",
        ),
        bigquery.SchemaField(
            "recommended_actions",
            "STRING",
            mode="REPEATED",
        ),
        bigquery.SchemaField(
            "priority",
            "STRING",
            mode="REQUIRED",
        ),
    ]

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition="WRITE_TRUNCATE",
    )

    print(
        f"Uploading {len(insights_df):,} AI insight rows "
        f"to BigQuery..."
    )

    load_job = client.load_table_from_dataframe(
        insights_df,
        table_ref,
        job_config=job_config,
    )

    load_job.result()

    table = client.get_table(table_ref)

    print("Upload successful!")
    print(f"Destination: {table_ref}")
    print(f"Rows in BigQuery: {table.num_rows:,}")


if __name__ == "__main__":

    insights_df = prepare_ai_insights()

    print("AI insights prepared successfully!")
    print(f"Rows: {len(insights_df):,}")

    upload_to_bigquery(insights_df)