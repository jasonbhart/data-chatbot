"""This script loads documents from Confluence and stores them in a ChromaDB instance."""
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import ConfluenceLoader

from chromadb.config import Settings

loader = ConfluenceLoader(
    url="https://springboard-edu.atlassian.net/wiki",
    username="jason.hart@springboard.com",
    api_key="ATATT3xFfGF0oYlQP-aceoHZUNeeOavWD0Y1r7eAiCBJPaaZj3MD7B2k-1-LXU0_sRkWdkedDRdl6Q0RHKX-5ASNNtJ9wNyvF5VSXIRnzfRMw4RAAtzu6jyTn-ItRdJq3Yboy_DOmZ_idW7lVkxqjDYVKmd1DNIQJ-9z4odN_5Wcf4d_JfHcwCI=6C861233",
    cloud=True
)
documents = loader.load(
    space_key="TechTeam",
    cql="type in (page,blogpost) order by lastmodified desc",
    include_attachments=False,
    limit=50,
    max_pages=1000)

for doc in documents:
    doc.page_content = doc.page_content.encode(
        encoding='ASCII', errors='ignore').decode()
    doc.metadata["system"] = "confluence"

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
    # Optional, defaults to .chromadb/ in the current directory
    persist_directory="../chromadb"
)
db = Chroma.from_documents(
    client_settings=chroma_settings,
    embedding=embedding_function,
    collection_name="confluence-techteam",
    documents=split_docs)
db.persist()
