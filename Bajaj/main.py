from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipeline.query_pipeline import get_rag_chain

import os
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

class QueryRequest(BaseModel):
    query: str

@app.post("/process-query")
async def process_query(request: QueryRequest):
    """
    Processes a natural language query and returns a structured decision.

    This endpoint takes a query, processes it through the RAG pipeline,
    and returns a structured JSON response with the decision, amount,
    and justification based on the source documents.
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=500, 
            detail="RAG chain is not available. Check server logs for initialization errors."
        )
        
    try:
        print(f"Processing query: {request.query}")
        
        result = rag_chain.invoke({"question": request.query})
        
        return result
    except Exception as e:
        print(f"An error occurred during query processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the LLM Document Processing API. Send a POST request to /process-query to ask a question."}