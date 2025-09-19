# main_bm25.py (a simplified version)

from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from typing import List

# Import our new BM25 pipeline
from rag_bm25 import BM25Processor 

# --- Pydantic Models (same as before) ---
class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

app = FastAPI()
processor = BM25Processor()

@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_submission(request: HackRxRequest):
    answers = processor.process_questions(
        doc_url=str(request.documents),
        questions=request.questions
    )
    return {"answers": answers}