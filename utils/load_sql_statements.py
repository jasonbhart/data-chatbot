"""Loads a local context file into a Chromedb vector database"""
# The context file is a JSONL text file with one line per example

import os
import json
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from chromadb.config import Settings

# find the .env file and load it
load_dotenv(find_dotenv())

# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# Path to dbt project root
DBT_ROOT_PATH = os.getenv('DBT_ROOT_PATH')

FILES = ['sql_statements.jsonl']

documents = []
for file in FILES:
    with open(Path(file).resolve(), "r", encoding="utf-8") as file:
        data = file.read()
    for line in data.splitlines():
        data = json.loads(line)
        title = data['question']
        metadata = {
            "system": "bigquery-sql",
            "title": title,
        }
        page_content = "Question: " + title + "\n" + \
            "SQL statement to answer: " + data['sql']
        page_content = page_content.encode(
            encoding='ASCII', errors='ignore').decode()
        documents.append(
            Document(page_content=page_content, metadata=metadata))

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
    collection_name="datateam",
    documents=documents)
db.persist()

print(f"Loaded {len(documents)} lines of example SQL statements into Chromadb")
