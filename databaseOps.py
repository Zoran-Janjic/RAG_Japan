import argparse
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from getEmbeddings import get_embedding_function
from langchain_community.vectorstores import Chroma

# Define the path for the Chroma database
CHROMA_PATH = "chroma_db_japanese_labor_law_1947"


def main():
    """
        Main function to handle the process of loading PDF documents, splitting them into chunks,
        and adding them to the Chroma database.
    """
    
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_pdf()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_pdf():
    """
        Load PDF documents using PyPDFLoader and return a list of Document objects.
    """
    loaders = [PyPDFLoader("LaborStandardsAct1947.pdf")]
    # Initialize an empty list 'docs' to store the documents loaded from the PDF file
    docs = []
    for file in loaders:
        docs.extend(file.load())
    print(f"Loaded {len(docs)} documents from PDF.")
    return docs


def split_documents(documents: list[Document]):
    """
        Split the documents into chunks using RecursiveCharacterTextSplitter.
    """
    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len,
                                                   is_separator_regex=False)
    splited_docs = text_splitter.split_documents(documents)
    print(f"Split into {len(splited_docs)} chunks.")
    return splited_docs


def add_to_chroma(chunks: list[Document]):
    """
        Add chunks to the Chroma database after calculating chunk IDs and checking for existing documents.
    """
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    """
        Calculate unique chunk IDs based on source, page, and chunk index.

    """

    # Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    """
        Clear the Chroma database by deleting the specified directory.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
