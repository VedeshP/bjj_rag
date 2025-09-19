import os
import asyncio
import aiohttp
import hashlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import List, Dict, Optional
import pickle
import json
from pathlib import Path

import torch
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

class OptimizedDocumentProcessor:
    """
    Optimized RAG pipeline with caching, parallel processing, and GPU acceleration.
    """
    def __init__(self):
        # Use GPU if available for embeddings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.llm = ChatGroq(
            model_name="llama3-8b-8192", 
            temperature=0,
            max_tokens=512  # Limit response length for speed
        )
        
        # Use a faster embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Cache directory
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[FAISS]:
        """Load vector store from cache if available."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Cache load error: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, vector_store: FAISS):
        """Save vector store to cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(vector_store, f)
        except Exception as e:
            print(f"Cache save error: {e}")
    
    async def _download_document(self, doc_url: str) -> bytes:
        """Async document download."""
        async with aiohttp.ClientSession() as session:
            async with session.get(doc_url) as response:
                response.raise_for_status()
                return await response.read()
    
    def _process_document(self, doc_content: bytes) -> List:
        """Process document content into chunks."""
        from tempfile import NamedTemporaryFile
        
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(doc_content)
            temp_file_path = temp_file.name
        
        try:
            # Load and chunk the document
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            
            # Optimized chunking parameters
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Smaller chunks for faster retrieval
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            
            return chunks
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    def _create_vector_store(self, chunks: List) -> FAISS:
        """Create vector store from chunks."""
        return FAISS.from_documents(
            documents=chunks, 
            embedding=self.embedding_model
        )
    
    def _create_optimized_rag_chain(self, vector_store: FAISS):
        """Create optimized RAG chain."""
        # Use more efficient retrieval
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 2,  # Fewer chunks for speed
                "fetch_k": 5,  # Fetch more initially, then filter
                "lambda_mult": 0.8  # Diversity in results
            }
        )
        
        # Optimized prompt template
        prompt_template = """
        Context: {context}
        
        Question: {question}
        
        Provide a concise answer based only on the context above.
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain
    
    async def process_questions_batch(self, doc_url: str, questions: List[str]) -> List[str]:
        """Process questions in parallel batches."""
        # Check cache first
        cache_key = self._get_cache_key(doc_url)
        vector_store = self._load_from_cache(cache_key)
        
        if not vector_store:
            # Download and process document
            print("Downloading document...")
            doc_content = await self._download_document(doc_url)
            
            print("Processing document...")
            chunks = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._process_document, doc_content
            )
            
            print("Creating vector store...")
            vector_store = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._create_vector_store, chunks
            )
            
            # Cache the vector store
            await asyncio.get_event_loop().run_in_executor(
                self.executor, self._save_to_cache, cache_key, vector_store
            )
        
        # Create RAG chain
        rag_chain = self._create_optimized_rag_chain(vector_store)
        
        # Process questions in parallel
        print("Processing questions in parallel...")
        tasks = [rag_chain.ainvoke(q) for q in questions]
        answers = await asyncio.gather(*tasks)
        
        return answers
    
    async def process_questions(self, doc_url: str, questions: List[str]) -> List[str]:
        """Main method to process questions."""
        try:
            return await self.process_questions_batch(doc_url, questions)
        except Exception as e:
            print(f"Error processing questions: {e}")
            raise
