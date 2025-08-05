
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from operator import itemgetter

from src.components.llm import get_groq_llm
from src.components.retriever import get_pinecone_vectorstore
from src.components.parser import load_documents_with_docling
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

class Justification(BaseModel):
    clause_text: str = Field(description="The exact text of the policy clause that justifies the decision.")
    reasoning: str = Field(description="A brief explanation of how this clause applies to the user's query.")

class FinalResponse(BaseModel):
    decision: str = Field(description="The final decision, either 'Approved' or 'Rejected'.")
    amount: int = Field(description="The payout amount if approved. Set to 0 if rejected.")
    justification: List[Justification] = Field(description="A list of justifications, mapping each decision point to a specific policy clause.")

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



def format_docs(docs):
    """Helper function to format retrieved documents for the prompt."""
    return "\n\n---\n\n".join(f"Clause Source: {doc.metadata.get('source', 'N/A')}\n\n{doc.page_content}" for doc in docs)

def get_rag_chain():
    """
    Builds and returns a RAG chain using a powerful Ensemble Retriever.
    This combines semantic search from Pinecone and keyword search from BM25.
    """
    
    docs_for_bm25 = load_documents_with_docling()

    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    bm25_retriever.k = 2 

    vectorstore = get_pinecone_vectorstore()
    pinecone_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, pinecone_retriever],
        weights=[0.5, 0.5]  
    )

    llm = get_groq_llm()
    json_parser = JsonOutputParser(pydantic_object=FinalResponse)
    rag_prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
        partial_variables={"format_instructions": json_parser.get_format_instructions()},
    )

    rag_chain = (
        {
            "context": itemgetter("question") | ensemble_retriever | format_docs,
            "question": itemgetter("question"),
        }
        | rag_prompt
        | llm
        | json_parser
    )
    
    return rag_chain