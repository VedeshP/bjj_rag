import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

API_TOKEN_BJJ = os.getenv("API_TOKEN_BJJ")

# PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

PINECONE_INDEX_NAME = "bajaj-insurance-policy"

DATA_PATH = os.path.join("data")