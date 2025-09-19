import os
import asyncio
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, HttpUrl
from typing import List
import uvicorn

from dotenv import load_dotenv
load_dotenv()

# Import the optimized processor
from optimized_rag_pipeline import OptimizedDocumentProcessor

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


# --- FastAPI Application ---

app = FastAPI(
    title="Optimized LLM Document Processing API",
    description="An optimized API to process documents and answer questions using a RAG pipeline.",
    version="2.0.0"
)

# Initialize the optimized document processor
processor = OptimizedDocumentProcessor()

@app.post(
    "/hackrx/run",
    response_model=HackRxResponse,
    summary="Process a document and answer questions",
    tags=["Processing"]
)
async def run_submission(
    request: HackRxRequest,
    api_key: str = Security(get_api_key)
):
    """
    Optimized endpoint that processes documents and answers questions efficiently.
    
    **Optimizations:**
    1. Async document downloading
    2. GPU acceleration for embeddings (if available)
    3. Document caching to avoid reprocessing
    4. Parallel question processing
    5. Optimized chunking and retrieval parameters
    """
    try:
        # Use the optimized processor
        answers = await processor.process_questions(
            doc_url=str(request.documents),
            questions=request.questions
        )
        return {"answers": answers}
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
