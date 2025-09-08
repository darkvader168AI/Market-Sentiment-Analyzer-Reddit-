
# config.py
# Load Reddit credentials from environment or Streamlit secrets.
import os
from dotenv import load_dotenv

# Load .env if present (not required in production if you set env vars)
load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_SECRET    = os.getenv("REDDIT_SECRET")
REDDIT_USERAGENT = os.getenv("REDDIT_USERAGENT", "sentiment-app/1.0")
# Do not raise here to allow demo mode; app checks presence and can run without live Reddit
