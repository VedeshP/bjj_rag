# src/pipeline/query_pipeline.py

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from operator import itemgetter

# Import our components, including the new reranker
from src.components.llm import get_groq_llm
from src.components.retriever import get_pinecone_vectorstore
from src.components.parser import load_documents_with_docling
from src.components.reranker import rerank_documents
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

# --- Pydantic Models for Output (No change) ---
class Justification(BaseModel):
    clause_text: str = Field(description="The exact text of the policy clause that justifies the decision.")
    reasoning: str = Field(description="A brief explanation of how this clause applies to the user's query.")

class FinalResponse(BaseModel):
    decision: str = Field(description="The final decision, either 'Approved' or 'Rejected'.")
    amount: int = Field(description="The payout amount if approved. Set to 0 if rejected.")
    justification: List[Justification] = Field(description="A list of justifications, mapping each decision point to a specific policy clause.")

# --- Prompt Template (No change) ---
RAG_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a specialized JSON-outputting insurance claim adjudication machine. Your sole purpose is to analyze the provided context and query, and return a single, valid JSON object with the decision. You must not add any conversational text, apologies, or explanations outside of the JSON structure. Your output must be ONLY the JSON object and nothing else.

You must base your decision STRICTLY on the policy clauses provided in the 'CONTEXT' section. Do not use any external knowledge or make assumptions.

Your output MUST conform to this JSON schema:
{format_instructions}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Here is the data for your analysis:

CONTEXT:
{context}

QUERY:
{question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


# --- Build the RAG Chain with Reranker ---

def format_docs(docs):
    return "\n\n---\n\n".join(f"Clause Source: {doc.metadata.get('source', 'N/A')}\n\n{doc.page_content}" for doc in docs)

def get_rag_chain():
    """
    Builds the full RAG chain with an Ensemble Retriever followed by a Gemini Reranker.
    """
    print("--- Building RAG Chain with Reranker ---")
    
    # 1. Load documents for the in-memory BM25 retriever
    print("Loading documents for BM25 retriever...")
    docs_for_bm25 = load_documents_with_docling()
    print("...documents loaded.")

    # 2. Initialize the Keyword Retriever (BM25)
    print("Initializing BM25 retriever...")
    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    print("...BM25 retriever initialized.")

    # 3. Initialize the Semantic Retriever (Pinecone)
    print("Initializing Pinecone retriever...")
    vectorstore = get_pinecone_vectorstore()
    pinecone_retriever = vectorstore.as_retriever()
    print("...Pinecone retriever initialized.")

    # 4. Initialize the Ensemble Retriever
    print("Initializing Ensemble Retriever...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, pinecone_retriever],
        weights=[0.5, 0.5]
    )
    print("...Ensemble retriever initialized.")

    # This new helper function orchestrates retrieval and reranking
    def retrieve_and_rerank(inputs):
        query = inputs["question"]
        
        # Step 1: Get a broad set of initial candidates (e.g., k=10)
        initial_docs = ensemble_retriever.invoke(query, k=10)
        
        # Step 2: Use our new component to rerank and get the best 2
        final_docs = rerank_documents(query=query, documents=initial_docs, top_k=2)
        
        # Step 3: Format the high-quality documents for the final prompt
        return format_docs(final_docs)

    # Initialize the other components
    llm = get_groq_llm()
    json_parser = JsonOutputParser(pydantic_object=FinalResponse)
    rag_prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
        partial_variables={"format_instructions": json_parser.get_format_instructions()},
    )

    # Build the final chain using our new helper function
    rag_chain = (
        {
            "context": retrieve_and_rerank,
            "question": itemgetter("question"),
        }
        | rag_prompt
        | llm
        | json_parser
    )
    
    print("--- RAG Chain with Reranker Built Successfully ---")
    return rag_chain