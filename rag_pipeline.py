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

load_dotenv()

class DocumentProcessor:
    """
    A class to handle the entire RAG pipeline from document URL to answers.
    """
    def __init__(self):
        # Initialize the LLM (Llama 3 via Groq) and the embedding model
        llm_model_options=["llama3-8b-8192"]
        self.llm = ChatGroq(model_name=llm_model_options[0], temperature=0)
        emb_model_options=["all-MiniLM-L6-v2", "all-MiniLM-L12-v2"]
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=emb_model_options[1],
            model_kwargs={'device': 'cpu'} # Use CPU for embedding
        )
        self.vector_store = None

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

    def _create_rag_chain(self):
        """Creates the RAG chain for question answering."""
        if not self.vector_store:
            raise ValueError("Vector store is not initialized. Call _create_vector_store first.")

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks

        # This prompt template is crucial for directing the LLM.
        prompt_template = """
        You are an expert assistant for answering questions about policy documents.
        Answer the question based ONLY on the following context.
        If the information is not in the context, say "I cannot find the information in the document."
        Be concise and directly answer the question.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # The RAG chain using LangChain Expression Language (LCEL)
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
    

# similarity search , mmr