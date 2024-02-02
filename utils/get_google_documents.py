"""This script loads documents from a Google Drive folder and stores them in a Chroma database."""
import os
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import GoogleDriveLoader
from chromadb.config import Settings

# find the .env file and load it
load_dotenv(find_dotenv())

# Path to GCP credentials file
GCP_CREDENTIALS_FILE = os.getenv('GCP_CREDENTIALS_FILE')
# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

loader = GoogleDriveLoader(
    folder_id="1QWxKw8sxx45m2YGKAblH_-GK9LbTljkp", # Data Team folder ID
    # Optional: configure whether to recursively fetch files from subfolders. Defaults to False.
    recursive=True,
    file_types=["document"],
    service_account_key=GCP_CREDENTIALS_FILE,
)
documents = loader.load()

for doc in documents:
    doc.page_content = doc.page_content.encode(encoding='ASCII', errors='ignore').decode()
    doc.metadata["system"] = "google-drive"

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)
split_docs = text_splitter.split_documents(documents)

embedding_function = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    max_retries=100,
    show_progress_bar=True
)

chroma_settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="../chromadb" # Optional, defaults to .chromadb/ in the current directory
)
db = Chroma.from_documents(
    client_settings=chroma_settings,
    embedding=embedding_function,
    collection_name="datateam",
    documents=split_docs)
db.persist()
