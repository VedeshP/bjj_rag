import os
import requests
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from langchain_community.document_loaders import UnstructuredURLLoader

from langchain_together import TogetherEmbeddings

# from langchain_openai import OpenAIEmbeddings

import pickle
import hashlib

load_dotenv()

class DocumentProcessor:
    """
    A class to handle the entire RAG pipeline from document URL to answers.
    """
    def __init__(self):
        # Initialize the LLM (Llama 3 via Groq) and the embedding model
        llm_model_options=["llama3-8b-8192"]
        self.llm = ChatGroq(model_name=llm_model_options[0], temperature=0)
        emb_model_options=["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "BAAI/bge-base-en-v1.5", "thenlper/gte-base"]
        # self.embedding_model = HuggingFaceEmbeddings(
        #     model_name=emb_model_options[1],
        #     model_kwargs={'device': 'cpu'} # Use CPU for embedding
        # )

        self.embedding_model = TogetherEmbeddings(
            # model="togethercomputer/m2-bert-80M-8k-retrieval"
            model="BAAI/bge-large-en-v1.5",

            chunk_size=32
        )
        # self.embedding_model= OpenAIEmbeddings(
        #     model="text-embedding-3-small"
        # )

        # self.vector_store = None


    # --- MINOR CHANGE IN THE HELPER ---
    def _get_vector_store_path(self, doc_url: str) -> str:
        """Creates a unique, safe folder name for a doc URL to use for caching."""
        url_hash = hashlib.md5(doc_url.encode()).hexdigest()
        # No longer needs .pkl, as it's a folder
        return f"vector_store_{url_hash}"


    def _load_and_chunk_document(self, doc_url: str) -> List:
        """Loads a PDF from a URL, saves it temporarily, and splits it into chunks."""
        try:
            # Download the PDF content
            # response = requests.get(doc_url)
            # response.raise_for_status()  # Raise an exception for bad status codes

            # Write the content to a temporary file
            # with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            #     temp_file.write(response.content)
            #     temp_file_path = temp_file.name

            # Load the PDF with PyPDFLoader
            # loader = PyPDFLoader(temp_file_path)
            # documents = loader.load()

            # changed the above as i am now using UnstructuredURLLoader
            loader = UnstructuredURLLoader(urls=[doc_url])
            documents = loader.load()

            # Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = text_splitter.split_documents(documents)
            return chunks
        finally:
            # Clean up the temporary file
            print("document loaded and chinked successfully.")
            # if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            #     os.remove(temp_file_path)


    def _create_vector_store(self, chunks: List):
        """Creates an in-memory FAISS vector store from document chunks."""
        print("Creating vector store from document chunks...")
        self.vector_store = FAISS.from_documents(documents=chunks, embedding=self.embedding_model)
        print("Vector store created successfully.")


    # def _create_rag_chain(self):
    #     """Creates the RAG chain for question answering."""
    #     if not self.vector_store:
    #         raise ValueError("Vector store is not initialized. Call _create_vector_store first.")

    #     # retriever = self.vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks
    #     retriever = self.vector_store.as_retriever(search_kwargs={"k": 5, "fetch_k": 20}, search_type="mmr")
    #     # This prompt template is crucial for directing the LLM.
    #     # prompt_template = """
    #     # You are an expert assistant for answering questions about policy documents.
    #     # Answer the question based ONLY on the following context.
    #     # If the information is not in the context, say "I cannot find the information in the document."
    #     # Be concise and directly answer the question.

    #     # CONTEXT:
    #     # {context}

    #     # QUES`TION:
    #     # {question}

    #     # ANSWER:
    #     # """

    #     # In rag_pipeline.py's _create_rag_chain method

    #     prompt_template = """
    #     You are a highly intelligent insurance policy assistant. Your goal is to answer questions from users clearly and concisely, based *only* on the context provided.

    #     - Provide a direct answer to the question.
    #     - **Keep your answer to a maximum of two or three sentences.**
    #     - Do not mention the context or the document in your answer. Just provide the answer as if you are an expert.
    #     - If the answer is not present in the provided context, you MUST respond with the single phrase: "The information is not available in the provided document."

    #     CONTEXT:
    #     {context}

    #     QUESTION:
    #     {question}

    #     ANSWER:
    #     """

    #     # prompt_template = """
    #     # You are a highly intelligent insurance policy assistant. Your goal is to answer questions from users clearly and concisely, based *only* on the context provided.

    #     # - Do not mention the context or the document in your answer. Just provide the answer directly as if you are an expert.
    #     # - If the answer is not present in the provided context, you MUST respond with the single phrase: "The information is not available in the provided document."

    #     # CONTEXT:
    #     # {context}

    #     # QUESTION:
    #     # {question}

    #     # ANSWER:
    #     # """

    #     # prompt_template = """
    #     # You are a meticulous assistant for a legal and compliance team. Your task is to answer questions based *strictly* on the provided context from a policy document.
    #     # - Analyze the context carefully.
    #     # - If the answer is explicitly stated, provide it directly and concisely.
    #     # - If the answer is not in the context, you MUST respond with "Based on the provided text, the information is not available." Do not use outside knowledge.
    #     # - If the context contains conflicting information, point out the ambiguity.

    #     # CONTEXT:
    #     # {context}

    #     # QUESTION:
    #     # {question}

    #     # ANSWER:
    #     # """
    #     prompt = ChatPromptTemplate.from_template(prompt_template)

    #     # The RAG chain using LangChain Expression Language (LCEL)
    #     rag_chain = (
    #         {"context": retriever, "question": RunnablePassthrough()}
    #         | prompt
    #         | self.llm
    #         | StrOutputParser()
    #     )
    #     return rag_chain


    def _create_rag_chain(self, vector_store): # <-- MODIFIED: accepts vector_store as an argument
        """Creates the RAG chain for question answering."""
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )
        prompt_template = """
        You are a highly intelligent insurance policy assistant. Your goal is to answer questions from users clearly and concisely, based *only* on the context provided.
        - Provide a direct answer to the question.
        - **Keep your answer to a maximum of two or three sentences.**
        - Do not mention the context or the document in your answer. Just provide the answer as if you are an expert.
        - If the answer is not present in the provided context, you MUST respond with the single phrase: "The information is not available in the provided document."

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain


    def process_questions(self, doc_url: str, questions: List[str]) -> List[str]:
        """
        Main method to orchestrate the RAG pipeline.
        """
        # 1. Load and Chunk the document
        chunks = self._load_and_chunk_document(doc_url)

        # 2. Create the Vector Store (Embeddings + FAISS)
        self._create_vector_store(chunks)

        # 3. Create the RAG Chain
        rag_chain = self._create_rag_chain()

        # 4. Process each question
        answers = []
        print("Processing questions...")
        for i, question in enumerate(questions):
            print(f"  - Answering question {i+1}/{len(questions)}: '{question[:50]}...'")
            answer = rag_chain.invoke(question)
            answers.append(answer)

        return answers
    


    def process_questions_new(self, doc_url: str, questions: List[str]) -> List[str]:
        """
        Main method with caching logic.
        It checks if a vector store for this URL already exists locally.
        If yes, it loads it. If no, it creates and saves it.
        """
        # --- MODIFIED: The path is now a FOLDER, not a .pkl file ---
        vector_store_path = self._get_vector_store_path(doc_url)

        if os.path.exists(vector_store_path):
            # --- MODIFIED: Use FAISS.load_local ---
            print(f"Loading cached vector store from: {vector_store_path}")
            # When loading, you must provide the live embedding model to reconnect with the static data
            vector_store = FAISS.load_local(
                vector_store_path, 
                self.embedding_model,
                # This new argument is required for security reasons
                allow_dangerous_deserialization=True 
            )
        else:
            # This part is mostly the same
            print("Creating new vector store...")
            chunks = self._load_and_chunk_document(doc_url)
            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=self.embedding_model
            )
            # --- MODIFIED: Use vector_store.save_local ---
            print(f"Saving new vector store to: {vector_store_path}")
            vector_store.save_local(vector_store_path)

        # The rest of the process is now much faster
        rag_chain = self._create_rag_chain(vector_store)

        answers = []
        print("Processing questions...")
        for i, question in enumerate(questions):
            print(f"  - Answering question {i+1}/{len(questions)}: '{question[:50]}...'")
            answer = rag_chain.invoke(question)
            answers.append(answer)

        return answers
# similarity search , mmr