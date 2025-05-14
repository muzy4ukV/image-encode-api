import os
import json
from google.oauth2 import service_account


_raw_key = os.environ.get("GCP_SA_KEY")
if not _raw_key:
    raise RuntimeError("Missing GCP_SA_KEY environment variable")

try:
    SERVICE_KEY = json.loads(_raw_key)
    CREDENTIALS = service_account.Credentials.from_service_account_info(SERVICE_KEY)
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON in GCP_SA_KEY: {e}")
