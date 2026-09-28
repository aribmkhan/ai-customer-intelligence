from google.cloud import bigquery
from google.oauth2 import service_account


PROJECT_ID = "novamart-customer-intelligence"


def get_bigquery_client():
    """Create a BigQuery client for local or Streamlit deployment."""

    try:
        import streamlit as st

        if "gcp_service_account" in st.secrets:
            credentials = service_account.Credentials.from_service_account_info(
                dict(st.secrets["gcp_service_account"])
            )

            return bigquery.Client(
                project=PROJECT_ID,
                credentials=credentials,
            )

    except (ImportError, FileNotFoundError):
        pass

    # Local development uses Google Application Default Credentials.
    return bigquery.Client(project=PROJECT_ID)