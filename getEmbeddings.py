import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os


def get_embedding_function():
    # Load the .env file
    load_dotenv()
    # Retrieve the API key from the environment variable
    google_api_key = os.getenv('GOOGLE_API_KEY')
    # print(f"google_api_key {google_api_key}")
    # Check if the API key is set
    if not google_api_key:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable")

    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    return embedding_function
