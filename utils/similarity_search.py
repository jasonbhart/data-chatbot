import os
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings

# find the .env file and load it
# this sets OpenAI and other service API keys
load_dotenv(find_dotenv())

# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
db = Chroma(persist_directory="./chromadb",
            embedding_function=embedding_function, collection_name="datateam")
docs = db.similarity_search("Question: What is the month-by-month trend in the number of student enrollments for each month of the current calendar year, from January 2023 up to and including November 14, 2023, without filtering based on the students' current status or the courses' active status? SQL statement to answer:", k=5, filter={
                            'system': 'bigquery-sql'})
# docs = db.get(
#     where={"$and": [{"system": {"$eq": "bigquery-schemas"}}, {'title': {"$eq": "dbt_models.prep_student_info"}}]},
#     include=["documents"],
# )
print(docs)
