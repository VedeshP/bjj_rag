import os
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, HttpUrl
from typing import List

from src.pipeline.query_pipeline import get_rag_chain

import uvicorn

from dotenv import load_dotenv
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(
    title="LLM Document Processing System",
    description="An API for querying insurance policy documents using a RAG pipeline.",
    version="1.0.0"
)


try:
    rag_chain = get_rag_chain()
except Exception as e:
    print(f"Failed to initialize RAG chain on startup: {e}")
    rag_chain = None

API_TOKEN = os.getenv("API_TOKEN_BJJ")

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    """
    Dependency function to validate the API key.
    The key is expected to be passed as 'Bearer <token>'.
    """
    if not api_key:
        raise HTTPException(
            status_code=403,
            detail="Could not validate credentials: No 'Authorization' header provided."
        )

    # Split the "Bearer <token>" string
    parts = api_key.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=403,
            detail="Invalid 'Authorization' header format. Expected 'Bearer <token>'."
        )

    token = parts[1]
    if token != API_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="Could not validate credentials: Invalid token."
        )
    return token


class HackRxRequest(BaseModel):
    """
    Defines the structure of the incoming request body.
    """
    documents: HttpUrl  # Pydantic validates this is a valid URL
    questions: List[str]


class HackRxResponse(BaseModel):
    """
    Defines the structure of the outgoing response body.
    """
    answers: List[str]


@app.get("/")
def read_root():
    return {"message": "Welcome to the LLM Document Processing API. Send a POST request to /process-query to ask a question."}


@app.post(
    "/hackrx/run",
    response_model=HackRxResponse,
    summary="Process a document and answer questions",
    tags=["Processing"]
)
async def run_submission(request: HackRxRequest,):
    """
    Process natural language queries and answer those queries
    """
    print(os.getenv("API_TOKEN_BJJ"))

    if rag_chain is None:
        raise HTTPException(
            status_code=500, 
            detail="RAG chain is not available. Check server logs for initialization errors."
        )
    
    try:
        print(f"Processing query")

        # Invoke the RAG chain with the provided document URL and questions
        result = rag_chain.invoke({
            "doc_url": request.documents,
            "questions": request.questions
        })

        return HackRxResponse(answers=result)
    except Exception as e:
        print(f"An error occurred during query processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))