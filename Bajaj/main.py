# main.py

import os
import time
import asyncio
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, HttpUrl
from typing import List
from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.pipeline.query_pipeline import get_rag_chain

# --- App and RAG Chain Initialization ---
app = FastAPI(
    title="HackRx LLM Document Q&A System",
    description="An API for querying pre-indexed documents using a RAG pipeline.",
    version="3.2.0" # Version up for the new batching logic
)

rag_chain = None
try:
    rag_chain = get_rag_chain()
except Exception as e:
    print(f"!!! CRITICAL: Failed to initialize RAG chain on startup: {e} !!!")

# --- Security and Pydantic Models (No change) ---
API_TOKEN = os.getenv("API_TOKEN_BJJ")
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    # ... (same security logic) ...
    if not API_TOKEN:
        raise HTTPException(status_code=500, detail="API_TOKEN_BJJ environment variable not set on the server.")
    if not api_key:
        raise HTTPException(status_code=403, detail="Authorization header is missing.")
    parts = api_key.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=403, detail="Invalid Authorization header format. Expected 'Bearer <token>'.")
    token = parts[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API token.")
    return token

class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

def format_result_to_string(result: dict) -> str:
    # ... (same formatting logic) ...
    if not result or "decision" not in result or "justification" not in result:
        return "An error occurred while processing this question."
    decision = result.get('decision', 'N/A')
    justifications = result.get('justification', [])
    if not justifications:
        return f"Decision: {decision}. No specific justification was found in the document."
    first_justification = justifications[0]
    clause = first_justification.get('clause_text', 'N/A').strip()
    reasoning = first_justification.get('reasoning', 'N/A').strip()
    return f"{reasoning} This is based on the clause: '{clause}'"


# --- API Endpoints with NEW Batching Logic ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the HackRx Document Q&A API. POST to /hackrx/run to process a document."}


@app.post(
    "/hackrx/run",
    response_model=HackRxResponse,
    summary="Process a document and answer questions",
    tags=["Processing"]
)
async def run_submission(request: HackRxRequest, api_key: str = Security(get_api_key)):
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain is not available due to a server startup error.")

    print(f"--- Received request with {len(request.questions)} questions. Processing in parallel batches. ---")
    start_time = time.time()
    
    # --- INTELLIGENT BATCHING CONFIGURATION ---
    BATCH_SIZE = 3  # Number of requests to send in parallel
    DELAY_BETWEEN_BATCHES = 1.0  # Seconds to wait between batches

    all_questions = request.questions
    final_answers = []

    for i in range(0, len(all_questions), BATCH_SIZE):
        batch_questions = all_questions[i:i + BATCH_SIZE]
        print(f"--- Processing batch {i//BATCH_SIZE + 1}: {len(batch_questions)} questions ---")
        
        # Create a list of async tasks for the current batch
        tasks = [rag_chain.ainvoke({"question": q}) for q in batch_questions]
        
        # Run the batch concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process and format the results for this batch
        for j, res in enumerate(results):
            question = batch_questions[j]
            if isinstance(res, Exception):
                error_message = f"Failed to process question '{question}': {res}"
                print(f"  - {error_message}")
                final_answers.append(error_message)
            else:
                formatted_answer = format_result_to_string(res)
                final_answers.append(formatted_answer)

        # If this is not the last batch, pause to respect rate limits
        if i + BATCH_SIZE < len(all_questions):
            print(f"--- Delaying for {DELAY_BETWEEN_BATCHES}s before next batch ---")
            await asyncio.sleep(DELAY_BETWEEN_BATCHES)

    end_time = time.time()
    print(f"--- Batch processing complete in {end_time - start_time:.2f} seconds ---")
    
    return HackRxResponse(answers=final_answers)