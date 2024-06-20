import os

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from getEmbeddings import get_embedding_function
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain_community.callbacks import get_openai_callback


def query_rag(query_text: str):
    """
        Query the RAG (Retrieval Augmented Generation) model to get a response based on the input query.

        Args:
            query_text (str): The input query text.

        Returns:
            str: The formatted response from the RAG model.
        """

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

    embedding_function = get_embedding_function()
    CHROMA_PATH = "chroma_db_japanese_labor_law_1947"
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the Chroma database for similar documents
    results = db.similarity_search_with_score(query_text, k=6)
    # print("results >>> ", results)

    # Extract Documents from results (assuming results is a list of (Document, score) tuples)
    documents = [doc for doc, _score in results]


    if query_text:
        # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        # prompt = prompt_template.format(context=documents, question=query_text)

        llm_response = get_llm_response(docs=documents, userQuery=query_text)

        # Format and return the response
        formatted_response = f"{llm_response}\n   > Sources: Labor Standards Act 1947"

        return formatted_response


def get_llm_response(docs, userQuery):
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    input_dict = {'input_documents': docs, 'question': userQuery}

    with get_openai_callback() as callback:
        llm_response = chain.invoke(input=input_dict)
        print(callback)

    return llm_response['output_text']
