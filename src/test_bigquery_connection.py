from google.cloud import bigquery


PROJECT_ID = "novamart-customer-intelligence"


def test_connection():
    """Test the connection to Google BigQuery."""

    client = bigquery.Client(project=PROJECT_ID)

    query = """
        SELECT
            COUNT(*) AS customer_count
        FROM
            `novamart-customer-intelligence.novamart_analytics.customers`
    """

    result = client.query(query).result()

    for row in result:
        print("BigQuery connection successful!")
        print(f"Customers in BigQuery: {row.customer_count:,}")


if __name__ == "__main__":
    test_connection()