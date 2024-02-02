"""Load dbt context from catalog and manifest JSON files into Chromadb."""
import os
import json
import re
from pathlib import Path
from typing import List, Union
from dotenv import load_dotenv, find_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from chromadb.config import Settings

# find the .env file and load it
load_dotenv(find_dotenv())

# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# Path to dbt project root
DBT_ROOT_PATH = os.getenv('DBT_ROOT_PATH')

class JSONLoader(BaseLoader):
    """Load dbt context from catalog and manifest JSON files."""
    def __init__(
        self,
        dbt_root_path: Union[str, Path],
        ):
        self.catalog_path = Path(dbt_root_path).joinpath("target", "catalog.json").resolve()
        self.manifest_path = Path(dbt_root_path).joinpath("target", "manifest.json").resolve()

    def load(self) -> List[Document]:
        """Load and return documents from the JSON file."""

        # Load JSON file
        with open(self.catalog_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            catalog_nodes = data['nodes']

        with open(self.manifest_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            manifest = data

        filtered_manifest = {}
        # Iterate through manifest 'nodes'
        for key, value in manifest['nodes'].items():
            if value['resource_type'] != 'model':
                continue
            if value['config']['enabled'] is False:
                continue
            if value['config']['docs']['show'] is False:
                continue
            model_long_name = key
            model_short_name = key.split('.')[-1]
            model_description = value['description']
            model_depends_on = []
            for node in value['depends_on']['nodes']:
                if node.startswith('model.springboard_dbt.'):
                    model_depends_on.append("dbt_models." + node.split('.')[-1])
            model_code = re.sub(r"\s+", " ", value['raw_code'])
            filtered_manifest[model_long_name] = {
                "model_name": model_long_name,
                "table_name": "dbt_models." + model_short_name,
                "description": model_description,
                "depends_on": model_depends_on,
                "source_code": model_code
            }

        dependents = {}
        for key, value in manifest['child_map'].items():
            deps = []
            model_long_name = key
            for item in value:
                if item.startswith('model.springboard_dbt.'):
                    table_name = item.split('.')[-1]
                    deps.append("dbt_models." + table_name)
            dependents[model_long_name] = ", ".join(deps)

        filtered_catalog = {}
        # Iterate through catalog 'nodes'
        for key, value in catalog_nodes.items():
            model_long_name = key
            model_columns = value['columns']
            filtered_catalog[model_long_name] = {
                "columns": model_columns
            }

        content = []
        for key, value in filtered_manifest.items():
            model_long_name = key
            metadata = dict(
                system = "dbt",
                title = value["table_name"],
                table_name = value["table_name"],
                model_name = value["model_name"],
                source="https://sandbox-219106.uc.r.appspot.com/#!/model/" + model_long_name
            )
            new_object = {}
            new_object[value["table_name"]] = {
                "table_description": value["description"],
                "table_depends_on": value["depends_on"],
                "table_source_code": value["source_code"]
            }
            if model_long_name in filtered_catalog:
                if "columns" in filtered_catalog[model_long_name]:
                    new_object["table_columns"] = filtered_catalog[model_long_name]['columns']
            if model_long_name in dependents:
                new_object["table_dependents"] = dependents[model_long_name]

            page_content = json.dumps(new_object, indent=2, sort_keys=False)
            page_content = page_content.encode(encoding='ASCII', errors='ignore').decode()
            print("Processed " + metadata['title'])
            content.append(Document(page_content=page_content, metadata=metadata))

        return content

loader = JSONLoader(
    dbt_root_path = DBT_ROOT_PATH
)

documents = loader.load()

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
