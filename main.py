import os
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, HttpUrl
from typing import List
import uvicorn


from dotenv import load_dotenv
load_dotenv()

from rag_pipeline import DocumentProcessor

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
    title="LLM Document Processing API",
    description="An API to process documents and answer questions using a RAG pipeline.",
    version="1.0.0"
)

# Initialize our document processor
# This object will be reused for requests, but the vector store will be rebuilt each time.
processor = DocumentProcessor()

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
    This endpoint receives a URL to a document and a list of questions.
    It will eventually process the document, find relevant clauses, and
    generate answers for each question.

    **Workflow:**
    1.  Download and parse the document.
    2.  For each question:
        a. Find relevant text chunks using semantic search.
        b. Generate an answer using an LLM with the retrieved context.
    3.  Return all answers in a structured JSON response.
    """

    # removed the below placeholder logic 
    # now adding actual RAG pipeline 

    try:
        # Call the RAG pipeline to get the answers
        answers = processor.process_questions(
            doc_url=str(request.documents),
            questions=request.questions
        )
        return {"answers": answers}
    except Exception as e:
        # Handle potential errors during processing
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # --- Placeholder Logic ---
    # In the next steps, we will replace this with the actual RAG pipeline.
    # This includes document loading, chunking, embedding, searching, and answer generation.
    # For now, it returns a correctly formatted response with dummy answers.
    # The response time acknowledgement is noted; we will optimize the RAG steps.

    # print(f"Received request for document: {request.documents}")
    # print(f"Received {len(request.questions)} questions.")

    # Create a placeholder response that matches the number of questions
    # placeholder_answers = [
    #     f"This is a placeholder answer for question #{i+1}."
    #     for i, q in enumerate(request.questions)
    # ]

    # The final response must match the HackRxResponse model
    # return {"answers": placeholder_answers}


