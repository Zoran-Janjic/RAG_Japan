import os

from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.prompts import ChatPromptTemplate
from getEmbeddings import get_embedding_function
import google.generativeai as genai
from databaseOps import get_chunks, get_llm_response


def query_rag(query_text: str):
    """
        Query the RAG (Retrieval Augmented Generation) model to get a response based on the input query.

        Args:
            query_text (str): The input query text.

        Returns:
            str: The formatted response from the RAG model.
        """

    # CHROMA_PATH = "chroma_db_japanese_labor_law_1947"

    # Define the template for the prompt
    PROMPT_TEMPLATE = """
    As a helpful assistant for those seeking insights into Japanese labor law, I'm here to provide clear and friendly answers.

    Context:
    {context}

    ---

    Your Question:
    {question}

    Answer:
    """

    # db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the Chroma database for similar documents
    # results = db.similarity_search_with_score(query_text, k=6)
    # print("results >>> ", results)

    # Get the context text from the search results
    # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    embedding_function = get_embedding_function()

    chunks = get_chunks()

    knowledge_base = FAISS.from_texts(chunks, embedding_function)

    if query_text:
        retrieved_docs = knowledge_base.similarity_search(query_text)

        llm_response = get_llm_response(docs=retrieved_docs, userQuery=query_text)

        # Format and return the response
        formatted_response = f"{llm_response}\n   > Sources: Labor Standards Act 1947"

        return formatted_response
