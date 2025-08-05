from langchain_groq import ChatGroq
from config import GROQ_API_KEY

def get_groq_llm():
    """
    Initializes and returns a ChatGroq LLM instance.

    This function configures the LLM for our specific needs:
    - Model: Llama3, a powerful and fast open-source model.
    - Temperature: 0, to ensure deterministic and consistent outputs,
      which is crucial for structured data generation like JSON.

    Returns:
        ChatGroq: An instance of the LangChain Groq chat model.
    """
    llm = ChatGroq(
        temperature=0,
        model_name="llama3-8b-8192",
        api_key=GROQ_API_KEY
    )
    return llm