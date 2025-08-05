import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    GOOGLE_API_KEY
)

EMBEDDING_DIMENSION = 768
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)

def embed_and_store_documents(documents, index_name=PINECONE_INDEX_NAME):
    """
    Embeds document chunks using Gemini and stores them in a Pinecone index.

    This function is idempotent:
    - It checks if the Pinecone index already exists.
    - If not, it creates a new index with the correct dimension.
    - It then uses LangChain's PineconeVectorStore to efficiently embed and
      upload the documents.

    Args:
        documents (list[Document]): A list of LangChain Document objects (from our parser).
        index_name (str): The name of the Pinecone index.
    """
    
    if index_name not in pc.list_indexes().names():
        print(f"Index not found. Creating a new index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,  
            metric="cosine",  
            spec=ServerlessSpec(
                cloud="aws", 
                region="us-east-1" 
            )
        )
        while not pc.describe_index(index_name).status['ready']:
            print("Waiting for index to be ready...")
            time.sleep(1)
        print("Index created successfully.")
    else:
        print("Index already exists. Proceeding to store documents.")

    print("Embedding documents and storing them in Pinecone...")
    vector_store = PineconeVectorStore.from_documents(
        documents,
        index_name=index_name,
        embedding=embeddings
    )
    print("...Documents have been successfully embedded and stored in Pinecone.")
    
    return vector_store