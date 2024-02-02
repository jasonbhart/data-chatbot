import os
import sys
from dotenv import load_dotenv, find_dotenv
import chromadb
from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings

# find the .env file and load it
# this sets OpenAI and other service API keys
load_dotenv(find_dotenv())

# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="../chromadb" # Optional, defaults to .chromadb/ in the current directory
))
embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
collection = client.get_collection(name="datateam", embedding_function=embedding_function)

if len(sys.argv) > 1:
    query = collection.delete(
        where={"system": sys.argv[1]},
    )
else:
    print("You must specify the system name to delete its entries.")
