"""Load BigQuery schemas and samples from the BigQuery INFORMATION_SCHEMA and TABLESAMPLE SYSTEM."""
import os
import json
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from sqlalchemy import create_engine, text
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import TokenTextSplitter
from chromadb.config import Settings

import include

# find the .env file and load it
load_dotenv(find_dotenv())

# Google Cloud project id
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
# BigQuery dataset name
BQ_DATASET_NAME = os.getenv('BQ_DATASET_NAME')
# Google Cloud service account key file
GCP_CREDENTIALS_FILE = os.getenv('GCP_CREDENTIALS_FILE')
# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# Path to dbt project root
DBT_ROOT_PATH = os.getenv('DBT_ROOT_PATH')

TABLES = []

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
            if 'ignore' in value['tags']:
                continue
            model_long_name = key
            model_short_name = key.split('.')[-1]
            if model_short_name not in include.tables:
                continue
            table_name = model_short_name
            TABLES.append(table_name)

engine = create_engine(
    f'bigquery://{GCP_PROJECT_ID}', credentials_path=GCP_CREDENTIALS_FILE)

documents = []

text_splitter = TokenTextSplitter(
    model_name="text-embedding-ada-002",
    chunk_size=8000
)

for table in TABLES:
    # Query the CREATE statement for the model
    with engine.connect() as connection:

        result = connection.execute(text(
            f'SELECT ddl FROM `{GCP_PROJECT_ID}.{BQ_DATASET_NAME}.INFORMATION_SCHEMA.TABLES` WHERE table_name = "{table}"'
        ))
        row = result.fetchone()
        if row is None:
            print("Skipping " + table + " because it has no DDL")
            continue
        DDL = str(row._mapping['ddl']).replace(
            f'{GCP_PROJECT_ID}.', '')  # pylint: disable=W0212
        if DDL is None:
            print("Warning: `" + table + "` has no DDL")
        result = connection.execute(text(
            # f'SELECT * FROM `{GCP_PROJECT_ID}.{BQ_DATASET_NAME}.{table}` ORDER BY RAND() LIMIT 3'
            f'SELECT * FROM `{GCP_PROJECT_ID}.{BQ_DATASET_NAME}.{table}` TABLESAMPLE SYSTEM '
            '(1 PERCENT) QUALIFY ROW_NUMBER() OVER (ORDER BY RAND()) <= 3'
        ))
        SAMPLES = '\n'.join(str(row._mapping)
                            for row in result)  # pylint: disable=W0212
        if SAMPLES is None:
            print("Warning: `" + table + "` has no sample data")

    TABLE_SCHEMA = DDL
    SAMPLE_DATA = SAMPLES

    for system in ['bigquery-schemas', 'bigquery-samples']:
        metadata = dict(
            system=system,
            title=f"{BQ_DATASET_NAME}.{table}",
            table_name=table,
        )
        if system == 'bigquery-schemas':
            if TABLE_SCHEMA is None:
                print("Skipping: `" + table + "` schemas")
                continue
            DOCUMENT = TABLE_SCHEMA

        elif system == 'bigquery-samples':
            if SAMPLE_DATA is None:
                print("Skipping: `" + table + "` samples")
                continue
            DOCUMENT = SAMPLE_DATA
        # print(DOCUMENT)
        DOCUMENT = DOCUMENT.encode(encoding='ASCII', errors='ignore').decode()
        SPLIT_DOCUMENT = text_splitter.split_text(DOCUMENT)
        if isinstance(SPLIT_DOCUMENT, list) and len(SPLIT_DOCUMENT) > 0:
            DOCUMENT = SPLIT_DOCUMENT[0]
        # Only use the first page of the document
        PAGE_CONTENT = DOCUMENT
        if system == 'bigquery-samples':
            print("Adding " + metadata['title'] + " to the document store")
        documents.append(
            Document(page_content=PAGE_CONTENT, metadata=metadata))

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
