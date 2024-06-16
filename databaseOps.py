import argparse
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from getEmbeddings import get_embedding_function
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI
from langchain_community.callbacks import get_openai_callback

# Define the path for the Chroma database
CHROMA_PATH = "chroma_db_japanese_labor_law_1947"


def main():
    """
        Main function to handle the process of loading PDF documents, splitting them into chunks,
        and adding them to the Chroma database.
    """

    # Check if the database should be cleared (using the --clear flag).
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--reset", action="store_true", help="Reset the database.")
    # args = parser.parse_args()
    # if args.reset:
    #     print("✨ Clearing Database")
    #     clear_database()

    # Create (or update) the data store.

    # add_to_chroma(chunks)


def get_chunks():
    textFileDocument = load_pdf()
    chunks = split_documents(textFileDocument)
    return chunks


def load_pdf():
    """
        Load the PDF document using PyPDFLoader and return a it.
    """
    pdf_reader = PdfReader("LaborStandardsAct1947.pdf")

    print(f"Loaded document from PDF.")

    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text


def split_documents(textFile):
    """
        Split the documents into chunks using RecursiveCharacterTextSplitter.
    """
    # Split the documents
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)

    chunks = text_splitter.split_text(textFile)

    print(f"Split into {len(chunks)} chunks.")

    return chunks


def get_llm_response(docs, userQuery):
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    input_dict = {'input_documents': docs, 'question': userQuery}

    with get_openai_callback() as callback:
        llm_response = chain.invoke(input=input_dict)
        print(callback)

    return llm_response['output_text']
