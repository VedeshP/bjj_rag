from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import (
    PINECONE_INDEX_NAME,
    GOOGLE_API_KEY
)

def get_pinecone_vectorstore():
    """
    Initializes and returns a connection to the existing Pinecone vector store.

    This vector store object is the foundation for creating retrievers. It
    knows how to communicate with our Pinecone index.

    Returns:
        PineconeVectorStore: A LangChain vector store object.
    """

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=GOOGLE_API_KEY
    )

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    
    return vectorstore