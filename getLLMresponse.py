import os

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from getEmbeddings import get_embedding_function
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain_community.callbacks import get_openai_callback
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import google.generativeai as genai


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
    As a helpful assistant for those seeking insights into Japanese labor law, I'm here to provide clear and friendly answers. Translate the provided question context to english first.

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

    documents = [doc for doc, _score in results]

    if query_text:
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        prompt = prompt_template.format(context=documents, question=query_text)

        llm_response = get_llm_response(prompt)

        # Format and return the response
        formatted_response = f"{llm_response}\n   > Sources: Labor Standards Act 1947"

        return formatted_response


def get_llm_response(prompt):
    load_dotenv()

    # OPEN AI OPTION
    # Retrieve the API key from the environment variable
    # openai_api_key = os.getenv('OPENAI_API_KEY')
    # llm = OpenAI(openai_api_key=openai_api_key)
    # chain = load_qa_chain(llm, chain_type="stuff")

    # with get_openai_callback() as callback:
    #     llm_response = chain.invoke(prompt)
    #     print(callback)

    # return llm_response['output_text']

    # GOOGLE GEMINI OPTION

    google_gemini_api_key = os.getenv('GOOGLE_API_KEY')

    if not google_gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")

    genai.configure(api_key=google_gemini_api_key)

    model_llm = genai.GenerativeModel("gemini-pro")
    print(f"Prompt >>> {prompt}")
    llm_response = model_llm.generate_content(prompt)

    # llm_model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
    #
    # chain = load_qa_chain(llm_model, chain_type="stuff")

    # input_dict = {
    #     'input_documents': prompt,
    #     'question': userQuery
    # }
    # print(input_dict)
    # Invoke the model with the correctly formatted input
    # llm_response = chain.invoke(input=prompt)
    # llm_response = llm_model(prompt)
    # print(llm_response)

    if llm_response.candidates:
        for candidate in llm_response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    extracted_text = part.text
                    print(f"Extracted Text: {extracted_text}")
                    return extracted_text

    # Extract text from the response
    # if 'candidates' in llm_response:
    #     for candidate in llm_response['candidates']:
    #         if 'content' in candidate and 'parts' in candidate['content']:
    #             for part in candidate['content']['parts']:
    #                 if 'text' in part:
    #                     print(f" GEMINI PRO llm_response IS: > > > {part['text']}")
    #                     return part['text']
    #
    # raise ValueError("Response does not contain the expected text")
