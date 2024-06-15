import os

from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from getEmbeddings import get_embedding_function
import google.generativeai as genai


def query_rag(query_text: str):
    """
        Query the RAG (Retrieval Augmented Generation) model to get a response based on the input query.

        Args:
            query_text (str): The input query text.

        Returns:
            str: The formatted response from the RAG model.
        """

    CHROMA_PATH = "chroma_db_japanese_labor_law_1947"

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

    # Prepare the Chroma database and embedding function
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the Chroma database for similar documents
    results = db.similarity_search_with_score(query_text, k=6)
    # print("results >>> ", results)

    # Get the context text from the search results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Create the prompt using the template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="phi3")
    response_text = model.invoke(prompt)

    # Get the sources for the search results
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    # Format and return the response
    formatted_response = f"{response_text}\n   > Sources: Labor Standards Act 1947"

    # print("formatted_response ", formatted_response)
    # print("response_text ", response_text)
    #
    return formatted_response
