# from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings


def get_embedding_function():
    embeddings = OpenAIEmbeddings(credentials_profile_name="default", region_name="use-east-1")
    return embeddings
