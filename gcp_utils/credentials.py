import os, json, base64
from google.oauth2 import service_account

encoded = os.getenv("GCP_SA_KEY_B64")
if not encoded:
    raise RuntimeError("Missing GCP_SA_KEY_B64")

decoded = base64.b64decode(encoded).decode("utf-8")
info = json.loads(decoded)

CREDENTIALS = service_account.Credentials.from_service_account_info(info)
