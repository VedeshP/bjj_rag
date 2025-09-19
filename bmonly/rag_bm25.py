# rag_bm25.py

import os
from dotenv import load_dotenv
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# The new import for our keyword search model
from rank_bm25 import BM25Okapi

load_dotenv()

class BM25Processor:
    def __init__(self):
        # We only need the LLM for the final generation step. No embedding model!
        self.llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)

    def _load_and_chunk_document(self, doc_url: str) -> List[str]:
        """
        Loads the document and splits it into text chunks.
        Returns a list of strings directly.
        """
        print("Loading and chunking document...")
        loader = UnstructuredURLLoader(urls=[doc_url])
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        chunks = text_splitter.split_documents(documents)
        
        # Filter out empty chunks and return only the text content
        return [chunk.page_content for chunk in chunks if chunk.page_content.strip()]

    def _create_rag_chain(self):
        """
        Creates a simpler RAG chain that accepts context directly.
        """
        prompt_template = """
        You are a highly intelligent insurance policy assistant. Your goal is to answer questions from users clearly and concisely, based *only* on the context provided.
        - Provide a direct answer to the question.
        - Keep your answer to a maximum of two or three sentences.
        - If the answer is not present in the provided context, you MUST respond with the single phrase: "The information is not available in the provided document."

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # This chain expects a dictionary with "context" and "question"
        rag_chain = prompt | self.llm | StrOutputParser()
        return rag_chain

    def process_questions(self, doc_url: str, questions: List[str]) -> List[str]:
        """
        Main method using BM25 for retrieval.
        """
        # 1. Load and chunk the document. This is very fast.
        chunks = self._load_and_chunk_document(doc_url)
        
        # 2. Create the BM25 index. This is also extremely fast.
        print("Creating BM25 index...")
        tokenized_corpus = [doc.split(" ") for doc in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # 3. Create the LLM chain.
        rag_chain = self._create_rag_chain()

        # 4. Process each question.
        answers = []
        print("Processing questions with BM25 retrieval...")
        for question in questions:
            # Tokenize the user's question
            tokenized_query = question.split(" ")
            
            # Retrieve the top N most relevant chunks using keyword search.
            # As you suggested, we fetch more chunks (e.g., 7) to improve our chances.
            top_chunks = bm25.get_top_n(tokenized_query, chunks, n=7)
            
            # Combine the retrieved chunks into a single context string
            context_string = "\n\n---\n\n".join(top_chunks)
            
            # 5. Invoke the chain with the retrieved context and question
            answer = rag_chain.invoke({
                "context": context_string, 
                "question": question
            })
            answers.append(answer)

        return answers