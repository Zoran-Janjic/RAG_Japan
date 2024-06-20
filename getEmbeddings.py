import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embedding_function():
    # Retrieve the API key from the environment variable
    google_api_key = os.getenv('GOOGLE_API_KEY')

    # Check if the API key is set
    if not google_api_key:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable")

    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embedding_function
