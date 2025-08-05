from src.components.parser import load_documents_with_docling
from src.components.vector_store import embed_and_store_documents

def run_ingestion_pipeline():
    """
    Executes the full data ingestion pipeline.
    
    1. Loads and parses documents using Docling.
    2. Embeds the documents using Gemini.
    3. Stores the embeddings in Pinecone.
    """
    print("Starting data ingestion pipeline...")
    
    try:
        documents = load_documents_with_docling()
    except Exception as e:
        print(f"Error during document loading/parsing: {e}")
        return

    if not documents:
        print("No documents were loaded. Aborting pipeline.")
        return

    try:
        embed_and_store_documents(documents)
    except Exception as e:
        print(f"Error during embedding/storage: {e}")
        return 
        
    print("Data ingestion pipeline completed successfully!")

if __name__ == "__main__":
    run_ingestion_pipeline()