import os
from langchain_docling import DoclingLoader

from config import DATA_PATH

def load_documents_with_docling():
    """
    Finds a PDF in the data directory and loads it using the integrated
    DoclingLoader from langchain-docling.

    This loader uses Docling's advanced AI models under the hood to perform
    structure-aware parsing and chunking directly, without needing a separate server.
    It returns a list of LangChain Document objects, where each object represents
    a logical "chunk" from the original document.

    Returns:
        list[Document]: A list of LangChain Document objects, ready for embedding.
    """
    pdf_file_path = None
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            pdf_file_path = os.path.join(DATA_PATH, filename)
            break

    if not pdf_file_path:
        raise FileNotFoundError("No PDF file found in the 'data' directory. Please add a sample PDF.")

    print(f"Loading document with DoclingLoader: {pdf_file_path}")

    loader = DoclingLoader(file_path=pdf_file_path)

    documents = loader.load()
    
    for doc in documents:
        doc.metadata = {
            "source": os.path.basename(pdf_file_path),
            "page": doc.metadata.get("page_num", -1) 
        }
    
    return documents