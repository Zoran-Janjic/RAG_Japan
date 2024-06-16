from langchain_openai import OpenAIEmbeddings
import os


def get_embedding_function():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model_name = "text-embedding-3-small"
    embeddings = OpenAIEmbeddings(api_key=openai_api_key, model=openai_model_name)
    return embeddings
