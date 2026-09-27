import pandas as pd
from google.cloud import bigquery


PROJECT_ID = "novamart-customer-intelligence"
DATASET_ID = "novamart_analytics"
TABLE_ID = "customer_segments"

SEGMENTS_FILE = "data/processed/final_customer_segments.csv"


def prepare_segments():
    """Prepare K-Means customer segment assignments for BigQuery."""

    df = pd.read_csv(SEGMENTS_FILE)

    segment_df = df[
        [
            "customer_id",
            "cluster",
            "segment_name",
        ]
    ].copy()

    return segment_df


def upload_to_bigquery(segment_df):
    """Upload customer segment assignments to BigQuery."""

    client = bigquery.Client(project=PROJECT_ID)

    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

    schema = [
        bigquery.SchemaField("customer_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("cluster", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("segment_name", "STRING", mode="REQUIRED"),
    ]

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition="WRITE_TRUNCATE",
    )

    print(f"Uploading {len(segment_df):,} rows to BigQuery...")

    load_job = client.load_table_from_dataframe(
        segment_df,
        table_ref,
        job_config=job_config,
    )

    load_job.result()

    table = client.get_table(table_ref)

    print("Upload successful!")
    print(f"Destination: {table_ref}")
    print(f"Rows in BigQuery: {table.num_rows:,}")


if __name__ == "__main__":
    segments = prepare_segments()

    print("Segment data prepared successfully!")
    print(f"Rows: {len(segments):,}")

    upload_to_bigquery(segments)