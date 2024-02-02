"""This script loads the table names from the dbt manifest.json file and imports them into Chromadb"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

import include

from chromadb.config import Settings


# find the .env file and load it
load_dotenv(find_dotenv())

# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# Path to dbt project root
DBT_ROOT_PATH = os.getenv('DBT_ROOT_PATH')

TABLES = []

# find the .env file and load it
load_dotenv(find_dotenv())

if DBT_ROOT_PATH:
    with open(Path(DBT_ROOT_PATH).joinpath("target", "manifest.json").resolve(), "r", encoding="utf-8") as file:
        data = json.load(file)
        manifest = data
        for key, value in manifest['nodes'].items():
            if value['resource_type'] != 'model':
                continue
            if value['config']['enabled'] is False:
                continue
            if value['config']['docs']['show'] is False:
                continue
            if value['config']['materialized'] != 'table':
                continue
            model_short_name = key.split('.')[-1]
            if model_short_name not in include.tables:
                continue
            table_name = "dbt_models." + model_short_name
            print("Added " + table_name)
            TABLES.append(table_name)

documents = []

for table in TABLES:
    metadata = dict(
        system="bigquery-tables",
        title=table,
    )
    page_content = table.encode(encoding='ASCII', errors='ignore').decode()
    documents.append(Document(page_content=page_content, metadata=metadata))

# encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
# token_count = len(encoding.encode(documents[0].page_content))

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
