# src/components/reranker.py

from langchain_core.documents import Document
from config import GOOGLE_API_KEY
from rank_llm.rerank.rankllm import PromptMode
from rank_llm.rerank.listwise.rank_gemini import SafeGenai
from rank_llm.data import Query, Candidate, Request

print("--- Initializing RankLLM Gemini Reranker ---")
try:
    reranker = SafeGenai(
        model="gemini-2.5-flash",
        context_size=8192,
        keys=[GOOGLE_API_KEY],
        prompt_mode=PromptMode.RANK_GPT,
    )
    print("--- RankLLM Gemini Reranker Initialized Successfully ---")
except Exception as e:
    print(f"!!! FAILED to initialize RankLLM Reranker: {e} !!!")
    reranker = None


def rerank_documents(query: str, documents: list, top_k: int = 2) -> list:
    """
    Reranks a list of documents based on their relevance to a query using RankLLM's SafeGenai.

    Args:
        query (str): The user's original query.
        documents (list): A list of LangChain Document objects from the retriever.
        top_k (int): The number of top documents to return after reranking.

    Returns:
        list: A new, re-ordered list of the top_k LangChain Document objects.
    """
    if reranker is None:
        print("--- Reranker is not available, returning top_k documents from original list as a fallback. ---")
        return documents[:top_k]
        
    if not documents:
        return []

    print(f"--- Reranking {len(documents)} documents for query: '{query}' ---")

    rank_llm_candidates = [
        Candidate(
            docid=f"doc_{i}",
            score=0.0,
            doc={"text": lc_doc.page_content, "metadata": lc_doc.metadata}
        )
        for i, lc_doc in enumerate(documents)
    ]
    
    request = Request(
        query=Query(text=query, qid="1"),
        candidates=rank_llm_candidates
    )

    try:
        batch_results = reranker.rerank_batch(requests=[request], top_k=top_k)
        rerank_results = batch_results[0]

        reranked_lc_docs = []
        for result in rerank_results.candidates:
            doc_content = result.doc["text"]
            doc_metadata = result.doc["metadata"]
            reranked_lc_docs.append(Document(page_content=doc_content, metadata=doc_metadata))
        
        print(f"--- Reranking complete. Returning top {len(reranked_lc_docs)} documents. ---")
        return reranked_lc_docs

    except Exception as e:
        print(f"!!! An error occurred during reranking: {e} !!!")
        print("--- Returning top_k documents from original list as a fallback. ---")
        return documents[:top_k]