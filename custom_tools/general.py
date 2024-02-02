"""Custom Tools for the Chatbot"""
import os
import functools
from typing import Dict
from pathlib import Path
import csv
import re
import json
import random

from dotenv import load_dotenv

import pandas as pd

from google.oauth2 import service_account
from google.cloud import bigquery
from google.cloud.bigquery.job import QueryJobConfig
from google.cloud.exceptions import NotFound, BadRequest
from google.cloud import storage

import tiktoken

from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.agents import Tool

from custom_tools.llms import gpt35, embedding_function
from custom_tools.google_cloud_storage import generate_signed_url
from custom_chains.fix_code_chain import fix_code_chain
from custom_chains.visualize_data_chain import visualize_data_chain
from custom_chains.imagine_tables_chain import imagine_tables_chain
from custom_chains.schema_search_chain import schema_search_chain
from custom_chains.summarize_learnings_chain import summarize_learnings_chain
from custom_chains.enrich_learnings_chain import enrich_learnings_chain
from custom_chains.translate_sql_chain import translate_sql_chain

# find the .env file and load it
# this sets OpenAI and other service API keys
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Google Cloud project id
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
# Google Cloud service account key file
GCP_CREDENTIALS_FILE = os.getenv('GCP_CREDENTIALS_FILE')
# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GCS_STORAGE_BUCKET = os.getenv('GCS_STORAGE_BUCKET')

VERBOSE = os.getenv('VERBOSE') == 'true'

if 'df' not in globals():
    df = pd.DataFrame()

if 'sql_statement' not in globals():
    sql_statement = ""

if 'learned_context' not in globals():
    learned_context = "None"

chromadb = Chroma(
    collection_name="datateam",
    persist_directory="./chromadb",
    embedding_function=embedding_function
)

gcp_credentials = service_account.Credentials.from_service_account_file(
    GCP_CREDENTIALS_FILE)
bq_client = bigquery.Client(
    credentials=gcp_credentials, project=GCP_PROJECT_ID)

gcs_client = storage.Client(
    credentials=gcp_credentials, project=GCP_PROJECT_ID)
gcs_bucket = gcs_client.bucket(GCS_STORAGE_BUCKET)


def tracefunc(func):
    """Decorates a function to show its trace."""

    @functools.wraps(func)
    def tracefunc_closure(*args, **kwargs):
        """The closure."""
        result = func(*args, **kwargs)
        print(f"{func.__name__}(args={args}, kwargs={kwargs}) => {result}")
        return result

    return tracefunc_closure

# setup utility functions


def list_all_tables():
    """List all tables in the database"""
    tables = chromadb.get(
        where={'system': 'bigquery-tables'},
        include=["documents"],
        limit=1000
    )
    return list(tables['documents'])


def fix_bq_date_function_parameter_order(string, function_name):
    """Fix the parameters of a BigQuery date (DATE_TRUNC) function"""
    pattern = fr"({function_name})\((.+?),\s+(.+?)\)\sAS"
    matches = re.findall(pattern, string, flags=re.IGNORECASE)
    for match in matches:
        first_argument = match[1]
        first_argument_modified = first_argument.strip(
            "\'").strip('\"').upper()
        second_argument = match[2]
        old_string = f"{function_name}({first_argument}, {second_argument})"
        new_string = f"{function_name}({second_argument}, {first_argument_modified})"
        string = string.replace(old_string, new_string)
    return string


def translate_calendar_functions(string):
    """Translate calendar functions to BigQuery functions"""
    pattern = fr"(YEAR|MONTH|DAY)\((.+?)\)\s"
    matches = re.findall(pattern, string, flags=re.IGNORECASE)
    for match in matches:
        first_argument = match[0]
        first_argument_modified = first_argument.strip(
            "\'").strip('\"').upper()
        second_argument = match[1]
        old_string = f"{first_argument}({second_argument})"
        new_string = f"EXTRACT({first_argument} FROM {first_argument_modified})"
        string = string.replace(old_string, new_string)
    return string


def process_data(input_dict: Dict):
    """Get the plan from the Streamlit session."""
    session_id = input_dict.pop('session_id')
    stage = str(input_dict.pop('stage'))
    vis_id = session_id + "_" + stage
    response = visualize_data_chain.invoke(input_dict)
    if response.visualization in ["Figure", "List", "Table"]:
        return (None, None, None)
    prepared_code = prepare_code_for_execution(response.code, vis_id)
    python = PythonAstREPLTool(locals={"df": input_dict['df']})
    imports = "import streamlit as st\nimport pandas as pd\nimport numpy as np\n"
    imports += "from matplotlib.ticker import ScalarFormatter\n"
    final_code = imports + prepared_code
    response = str(python.run(final_code))
    fix_code_iterations = 0
    while "Error:" in response and fix_code_iterations < 3:
        if VERBOSE:
            print("\n*** Fixing the visualization code due to error...\n")
            print(input_dict['df'].info(verbose=True))
            print(final_code)
            print(response)
            print("\n")
        final_code = fix_code_chain.invoke({
            "code": final_code,
            "error": response
        }).fixed_code
        fix_code_iterations += 1
        response = str(python.run(final_code))

    vis_url = None
    csv_url = None
    try:
        # upload the image to Google Cloud Storage
        blob = gcs_bucket.blob(f"{vis_id}.png")
        blob.upload_from_filename(f"temp/{vis_id}.png")
        # generate a signed url for the image
        vis_url = generate_signed_url(
            GCP_CREDENTIALS_FILE,
            GCS_STORAGE_BUCKET,
            f"{vis_id}.png"
        )
        # delete the image from the local filesystem
        os.remove(f"temp/{vis_id}.png")

        # upload the dataframe (in the form of a CSV) to Google Cloud Storage
        blob = gcs_bucket.blob(f"{vis_id}.csv")
        blob.upload_from_string(input_dict['df'].to_csv(index=False))
        csv_url = generate_signed_url(
            GCP_CREDENTIALS_FILE,
            GCS_STORAGE_BUCKET,
            f"{vis_id}.csv"
        )
    except Exception as error:
        print(error)

    return (final_code, vis_url, csv_url)


def prepare_code_for_execution(response, vis_id):
    """Prepare the code for execution and display in Streamlit app"""
    # vis_code = re.sub(r"plt\.show\(.*?\)", "st.pyplot(plt.gcf())", response)
    vis_code = re.sub(r"plt\.show\(.*?\)",
                      f"plt.savefig('temp/{vis_id}.png')", response)
    vis_code = vis_code.replace("\n ", "\n")
    if "axis.ticklabel_format" not in vis_code:
        set_axis_code = (
            "\n"
            "axis_count=0\n"
            "current_axis = plt.gca()\n"
            "for axis in [current_axis.xaxis, current_axis.yaxis]:\n"
            "  if axis.get_scale() == 'linear' and isinstance(axis.get_major_formatter(), ScalarFormatter):\n"
            "    axis.get_major_formatter().set_scientific(False)\n"
            "    axis.get_major_formatter().set_useOffset(False)\n"
            "    if axis_count == 0:\n"
            "      current_axis.set_xlim(left=0)\n"
            "    else:\n"
            "      current_axis.set_ylim(bottom=0)\n"
            "  axis_count += 1\n"
            "\n"
        )
        # idx = vis_code.find("st.pyplot")
        idx = vis_code.find("plt.savefig")
        vis_code = vis_code[:idx] + set_axis_code + vis_code[idx:]

    return vis_code


def format_sql_statement(string: str) -> str:
    """Format the SQL statement"""
    string = string.replace("\n", " ")
    string = string.replace("SELECT", "SELECT\n  ")
    # string = string.replace("FROM", "\nFROM\n  ")
    string = string.replace("WHERE", "\nWHERE\n  ")
    string = string.replace("GROUP BY", "\nGROUP BY\n  ")
    string = string.replace("HAVING", "\nHAVING\n  ")
    string = string.replace("ORDER BY", "\nORDER BY\n  ")
    string = string.replace("LIMIT", "\nLIMIT")
    string = "```sql\n" + string + "\n```"
    return string


def count_num_tokens(string: str) -> int:
    """Returns the number of tokens"""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def summarize_learnings(query: str, work_log: str = "", date: str = "") -> str:
    """Use LLM to summarize the learnings from the work log"""

    summary = summarize_learnings_chain.invoke({
        "work_log": work_log,
        "query": query,
        "date": date
    })
    summary = str(summary).replace("#", "")
    return summary


def enrich_learnings(
        query: str,
        work_log: str = "",
        existing_summary: str = "",
        date: str = "",
        sql_query: str = "") -> str:
    """Use LLM to summarize the learnings from the work log"""

    if sql_query:
        sql_query_learned_context = "- The last SQL query executed without errors was: " + sql_query + "\n"
    else:
        sql_query_learned_context = ""
    summary = enrich_learnings_chain.invoke({
        "work_log": work_log,
        "query": query,
        "learned_context": existing_summary,
        "sql_query": sql_query_learned_context,
        "date": date
    })
    summary = str(summary).replace("#", "")
    return summary


# def move_focus_to_top():
#     # inspect the html to determine control to specify to receive focus (e.g. text or textarea).
#     st.components.v1.html(
#         f"""
#             <script>
#                 var textarea = window.parent.document.querySelectorAll("textarea[type=textarea]");
#                 for (var i = 0; i < textarea.length; ++i) {{
#                     textarea[i].focus();
#                 }}
#             </script>
#         """,
#     )

# def sticky_header():

#     # make header sticky.
#     st.markdown(
#         """
#             <div class='fixed-header'/>
#             <style>
#                 div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
#                     position: sticky;
#                     top: 2.875rem;
#                     background-color: white;
#                     z-index: 999;
#                 }
#                 .fixed-header {
#                     border-bottom: 1px solid black;
#                 }
#             </style>
#         """,
#         unsafe_allow_html=True
#     )


general_retrievalqa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=gpt35,
    chain_type="stuff",
    retriever=chromadb.as_retriever(
        search_kwargs={"k": 5, 'filter': {'system': 'google-drive'}}),
    return_source_documents=False,
    chain_type_kwargs={
        "verbose": VERBOSE
    },
)


def general_retrievalqa_func(query: str) -> str:
    """Run the retrievalqa chain on a query and return the outputs as a json string"""
    chain_output = general_retrievalqa_chain(
        {general_retrievalqa_chain.question_key: query},
        return_only_outputs=True
    )
    output = chain_output["answer"]
    if len(chain_output["sources"]) > 5:
        output += "\n\nSources: " + chain_output["sources"]

    return output


def data_table_list_func(_) -> str:
    """List all tables in the database"""

    if VERBOSE:
        print("\n*** Getting all tables from the vectorstore")
    all_tables = list_all_tables()
    response = {}
    response["fully_qualified_table_names"] = all_tables
    return json.dumps(response)


def data_table_search_func(query: str) -> str:
    """Do a similarity search on the vectorstore using the query"""

    if VERBOSE:
        print(
            f"\n*** Doing a similarity search on the vectorstore using the query: {query}")
    result = chromadb.similarity_search(
        query, k=5, filter={'system': 'bigquery-tables'})
    for idx, _ in enumerate(result):
        result[idx] = result[idx].page_content
    response = {}
    response["fully_qualified_table_names"] = result
    return json.dumps(response)


def data_table_schema_lookup_func(table: str) -> str:
    """Get the schema of a table in the database from the vectorstore"""

    if VERBOSE:
        print(
            "\n*** Getting the schema of a table in the database from "
            f"the vectorstore using the table: {table}"
        )
    if "," in table:
        tables = csv.reader([table])
        return (
            "Only one table name at a time can be processed with this command. "
            "Please try again with only one table name."
        )
    else:
        tables = [table]
    documents_string = ""
    for item in tables:
        schema = chromadb.get(
            where={
                "$and": [{"system": {"$eq": "bigquery-schemas"}}, {'title': {"$eq": item}}]},
            include=["documents"],
        )
        result = schema['documents']

        if len(result) > 0:
            documents_string += "\n".join(result)

    if len(documents_string) > 0:
        return documents_string
    return "No table schemas found. The table(s) may not exist in the database."

# data_table_schema_search_retriever = chromadb.as_retriever(
#     search_kwargs = {"k":5, 'filter': {'system': 'bigquery-schemas'}}
# )
# data_table_schema_search_prompt = ChatPromptTemplate.from_template(DATA_TABLE_SCHEMA_SEARCH)

# data_table_schema_search_chain = (
#     {
#         "context": data_table_schema_search_retriever,
#         "question": RunnablePassthrough()
#     } |
#     data_table_schema_search_prompt |
#     gpt35 |
#     StrOutputParser())

# def data_table_schema_search_func(query:str) -> str:
#     """Run the schema retrieval chain on a query"""

#     if VERBOSE:
#         print(f"\n*** Running the schema retrieval chain on the query: {query}")
#     return data_table_schema_search_chain.invoke(query)


def get_similar_sql_queries_func(query: str):
    """Get similar SQL queries using the query"""

    augmented_query = "Question: " + query + "\nSQL statement to answer: "

    if VERBOSE:
        print(
            f"\n*** Getting similar SQL queries.")

    similar_sql_queries = chromadb.similarity_search(
        augmented_query, k=5, filter={'system': 'bigquery-sql'})

    for idx, _ in enumerate(similar_sql_queries):
        # Add logic to filter based on similarity score
        similar_sql_queries[idx] = similar_sql_queries[idx].page_content
    similar_sql_queries_str = "\n\n".join(similar_sql_queries)

    return similar_sql_queries_str


def get_similar_table_schemas_func(query):
    """Get similar table schemas using the query"""
    if isinstance(query, str):
        chain = schema_search_chain
        payload = {
            "query": query,
        }
    elif isinstance(query, dict):
        chain = imagine_tables_chain
        payload = query
    else:
        return "Error- Invalid query. Please enter a valid query."

    if VERBOSE:
        print(
            f"\n*** Getting similar table schemas using the query: {payload['query']}")
    tables = chain.invoke(payload)
    if isinstance(tables.table_schemas_converted_to_create_table_statements, list):
        tables_str = "\n".join(
            tables.table_schemas_converted_to_create_table_statements)
    else:
        tables_str = tables.table_schemas_converted_to_create_table_statements
    if VERBOSE:
        print(tables_str)
    similar_table_scemas = chromadb.similarity_search(
        tables_str, k=3, filter={'system': 'bigquery-schemas'})
    for idx, _ in enumerate(similar_table_scemas):
        similar_table_scemas[idx] = similar_table_scemas[idx].page_content
    similar_table_scemas_str = "\n".join(similar_table_scemas)
    similar_table_scemas_str = similar_table_scemas_str.replace(
        f"{GCP_PROJECT_ID}.", "")
    return similar_table_scemas_str


def data_table_samples_lookup_func(table: str) -> str:
    """Get sample data of a table in the database from the vectorstore"""

    if VERBOSE:
        print(
            "\n*** Getting sample data of a table in the database from "
            f"the vectorstore using the table: {table}"
        )
    if "," in table:
        return (
            "Only one table name at a time can be processed with this command. "
            "Please try again with only one table name."
        )
    if "." not in table:
        return (
            "You must use a fully-qualified table name (like dbt_models.$table) with this command. "
        )
    schema = chromadb.get(
        where={"$and": [{"system": {"$eq": "bigquery-samples"}},
                        {'title': {"$eq": table}}]},
        include=["documents"],
    )
    result = schema['documents']
    if len(result) > 0:
        documents_string = "\n".join(result)
        return documents_string
    else:
        return (
            "No table schemas found for this table. "
            "The table may not exist in the database."
        )


def data_table_metadata_lookup_func(table: str) -> str:
    """Get the metadata of a table in the database from the vectorstore"""

    if VERBOSE:
        print(
            "\n*** Getting the metadata of a table in the database from "
            f"the vectorstore using the table: {table}"
        )
    if "," in table:
        return (
            "Only one table name at a time can be processed with this command. "
            "Please try again with only one table name."
        )
    if "." not in table:
        return (
            "You must use a fully-qualified table name (like dbt_models.$table) with this command. "
        )
    metadata = chromadb.get(
        where={"$and": [{"system": {"$eq": "dbt"}},
                        {'title': {"$eq": table}}]},
        include=["documents"],
    )
    result = metadata['documents']
    if len(result) > 0:
        documents_string = "\n".join(result)
        return documents_string
    else:
        return (
            "No table metdata found for this table. "
            "The table may not exist in the database."
        )


def query_sql_db(query: str) -> str:
    """Query the SQL database and return the results as a json string"""

    if VERBOSE:
        print(f"\n*** Querying the SQL database using the query: {query}")

    if "." not in query:
        return (
            "You must use a fully-qualified table name (like dbt_models.$table) in your SQL statement. "
        )
    if " df " in query:
        return (
            "Do not use this command to query or interact with the dataframe. "
            "Use the 'Pandas Dataframe Tool' instead."
        )
    global df, sql_statement
    retry_query = False
    sql_error = ""
    try:
        query_job = bq_client.query(query, job_config=QueryJobConfig())
        df = query_job.to_dataframe(
            progress_bar_type=None
        )
        df = pd.DataFrame(df).convert_dtypes()
    except BadRequest as error:
        sql_error = str(error)

        if "DATE_TRUNC" in query:
            if ("A valid date part name is required" in sql_error or
                    "Found invalid date part argument function call syntax" in sql_error):
                query = fix_bq_date_function_parameter_order(
                    query, "DATE_TRUNC")
                retry_query = True
                sql_error = (
                    "Invalid use of DATE_TRUNC() function. Use the following syntax: "
                    "DATE_TRUNC(<date_expression>, <date_part>)\nWhere <date expression> "
                    "is a date or timestamp column and <date_part> is one of the following: "
                    "YEAR, MONTH, DAY, WEEK, QUARTER."
                )
        elif "UNION SELECT" in query:
            query = query.replace("UNION SELECT", "UNION ALL SELECT")
            retry_query = True
            sql_error = "Invalid use of UNION syntax. Use 'UNION ALL' or 'UNION DISTINCT' instead."
        elif "Function not found: YEAR; Did you mean" in sql_error:
            query = translate_calendar_functions(query)
            retry_query = True
            sql_error = "Invalid function name. Use EXTRACT(YEAR FROM ...). "
        elif "Function not found: MONTH; Did you mean" in sql_error:
            query = translate_calendar_functions(query)
            retry_query = True
            sql_error = "Invalid function name. Use EXTRACT(MONTH FROM ...). "
        elif "Function not found: DAY; Did you mean" in sql_error:
            query = translate_calendar_functions(query)
            retry_query = True
            sql_error = "Invalid function name. Use EXTRACT(DAY FROM ...). "
        else:
            if VERBOSE:
                print("\n*** Fixing the SQL query due to error...\nError: ")
                print(sql_error)
                print("\n")
            query = translate_sql_chain.invoke(
                {"sql_statement": query, "error": error}).sql_statement
            retry_query = True
    except NotFound as error:
        sql_error = str(error)
        if "Not found: Table" in sql_error:
            sql_error = sql_error.replace(
                f"Not found: Table {GCP_PROJECT_ID}:",
                "Table not found. Verify you have fully qualified (like dbt_models.$table) "
                "the table name and that the table exists then try again. "
            )
            retry_query = False

    if retry_query:
        if VERBOSE:
            print(f"\n*** Retrying query using revised query: {query}")
        try:
            query_job = bq_client.query(query, job_config=QueryJobConfig())
            df = query_job.to_dataframe(
                progress_bar_type=None
            )
            df = pd.DataFrame(df).convert_dtypes()
            sql_error = ""
            sql_statement = query
        except BadRequest as error:
            sql_error = str(error)
        except NotFound as error:
            sql_error = str(error)
            if "Not found: Table" in sql_error:
                sql_error = sql_error.replace(
                    f"Not found: Table {GCP_PROJECT_ID}:",
                    "Table not found. Verify you have fully qualified (like dbt_models.$table) "
                    "the table name and that the table exists then try again. "
                )

    if sql_error:
        sql_error = sql_error.replace(f"Dataset {GCP_PROJECT_ID}:", "")
        if "must be qualified with a dataset (e.g. dataset.table)" in sql_error:
            sql_error = sql_error.replace("must be qualified with a dataset (e.g. dataset.table)",
                                          "not found. Remember you must use only fully-qualified "
                                          "table names. Verify the table exists then try again")
        elif "Unrecognized name: " in sql_error and "JOIN" in sql_error:
            sql_error = sql_error.replace("Unrecognized name: ",
                                          "Improper JOIN Syntax. Use table aliases when "
                                          "specifying JOIN conditions. ")
        elif "Unrecognized name: dbt_models" in sql_error:
            sql_error = sql_error.replace("Unrecognized name: dbt_models",
                                          "Do not use fully-qualified column names in your query. ")
        elif "Unrecognized name: " in sql_error:
            sql_error = sql_error.replace("400 Unrecognized name: ",
                                          "Field not found: ")
        elif "No matching signature for" in sql_error:
            sql_error = sql_error.replace("No matching signature for",
                                          "Invalid field type use in ")
        elif "not found inside" in sql_error:
            sql_error = sql_error.replace("not found inside",
                                          ": field doesn't exist in this table. Verify the "
                                          "table schema and try again. ")
        elif "Failed to parse input string" in sql_error and "PARSE_DATE" in query:
            sql_error = "Invalid value used in PARSE_DATE() function. "
        elif "Invalid field type use in  operator" in sql_error:
            sql_error = ("Incompatible field types used in comparison operator. "
                         "You must use the same field types in the comparison. ")
        elif " Table \"df\" not found." in sql_error or " FROM df " in query:
            sql_error = (
                "Do not use this command to query or interact with the dataframe. "
                "Use the 'Pandas Dataframe Tool' instead."
            )
        return "Error- " + sql_error

    sql_statement = query
    df_len = len(df)
    if df_len == 1:
        json_string = df.to_json(orient="records")
        if df.iat[0, 0] == 0:
            return (
                f"Warning- Result of database query is: {json_string}. "
                "This is suspicious and may indicate a false assumption in the SQL. "
                "Check to make sure criteria used in the query are using valid values."
            )
        else:
            return f"Success: Result of database query is: {json_string}"
    elif df_len > 1:
        head = df.head()
        return (
            "Success- Results of database query written to dataframe `df`.\n"
            "When df.head() is run it yields the following results:\n" + str(head) + "\n"
            "If needed, use the 'Pandas Dataframe Tool' to query or manipulate the dataframe."
        )
    else:
        return (
            "Warning- No rows returned from database query. "
            "This is suspicious and may indicate a false assumption in the SQL. "
            "Check to make sure criteria used in the query are using valid values."
        )


def python_func(query: str) -> str:
    """Run a python command and return the output as a string"""
    global df
    original_df = df.copy()
    if query:
        if len(df) == 0:
            return "Error- dataframe `df` is empty. Please run a valid SQL query first."
        elif "plt.show(" in query:
            return (
                "Error- Do not plot or graph the data. Use the `Final Answer` "
                "command when done processing dataframe."
            )
        elif ".plot(" in query:
            return (
                "Error- Do not plot or graph the data. Use the `Final Answer` "
                "command when done processing dataframe."
            )
        python = PythonAstREPLTool(globals={"df": df})
        if not query.startswith("df=") and not query.startswith("df = "):
            query = "df=" + query
        query = "import pandas as pd\n" + query
        output = python.run(query)
        if df.equals(original_df):
            if len(output) > 0:
                if "Error:" in output:
                    return "Error- " + output
                else:
                    return ("Warning- Dataframe `df` was NOT modified. If you meant to modify the "
                            "dataframe with this expression, modify the expression to assign the "
                            "result back to the original dataframe and try again.\n"
                            "Expression result:\n") + str(output)
            else:
                return "Warning- Dataframe `df` was NOT modified and expression resulted in no output."
        else:
            return ("Success- Command executed successfully and Dataframe `df` contains the resulting dataframe.\n"
                    "Expression result:\n") + str(output) + "\nResults of running `df.head()`:\n" + str(df.head())
    else:
        return (
            "Error- Input to this command cannot be blank. Enter a valid Python Pandas syntax "
            "to use with Dataframe `df` or use the command 'Final Answer' if you have completed "
            "servicing the request."
        )


def python_read_func(query: str) -> str:
    """Run a python command and return the output as a string"""
    global df
    if query:
        if len(df) == 0:
            return "Error- dataframe `df` is empty. Please run a valid SQL query first."
        elif "plt.show(" in query:
            return (
                "Error- Do not plot or graph the data. Use the `Final Answer` "
                "command when done processing dataframe."
            )
        elif ".plot(" in query:
            return (
                "Error- Do not plot or graph the data. Use the `Final Answer` "
                "command when done processing dataframe."
            )
        python = PythonAstREPLTool(locals={"df": df})
        query = "import pandas as pd\n" + query
        output = python.run(query)
        if len(output) > 0:
            return ("Success- Dataframe `df` was NOT modified. Here is the result of the "
                    "expression:\n") + str(output)
        else:
            return "Warning- Dataframe `df` was NOT modified and expression resulted in no output."
    else:
        return (
            "Error- Input to this command cannot be blank. Enter a valid Python Pandas syntax "
            "to use with Dataframe `df` or use the command 'Final Answer' if you have completed "
            "servicing the request."
        )


def human_tool_func(query: str) -> str:
    """Return the input query as a string"""
    global df, sql_statement
    if not df.empty:
        df = df.iloc[0:0]
    if sql_statement:
        sql_statement = ""
    return query

# load the agent tools


def get_toolkit():
    """Custom toolkit for the agent"""

    tools = [
        Tool(
            name="Database Table Name Search",
            func=data_table_search_func,
            description=(
                "Useful for finding tables with similar names to a given question. "
                "Input should be a detailed, self-contained question "
                "and the output will be a json-serialized list of relevant "
                "fully-qualified table names"
            )
        ),
        Tool(
            name="Database Full Table List",
            func=data_table_list_func,
            description=(
                "Useful to list all available tables. "
                "Output is a json serialized list of all fully-qualified table names"
            )
        ),
        Tool(
            name="Database Table Schema Lookup",
            func=data_table_schema_lookup_func,
            description=(
                "Useful to view the full schema of a single table in the database. "
                "It displays all column names and data types, and available descriptions. "
                "Make sure the table exists in the database before using this command. "
                "Always use fully-qualified table names with this command."
                "Input should be a single fully-qualified table name. "
                "Output is a string with the CREATE TABLE statement"
            )
        ),
        Tool(
            name="Database Table Schema Search",
            func=get_similar_table_schemas_func,
            description=(
                "Useful to search for possibly related tables in the database by using a "
                "similarity comparison between the question and the table schema. "
                "It displays table names, descriptions, and a curated list of possibly "
                "related column names and data types. "
                "Input should be a detailed question with as much context as possible."
                "Output is a string with the CREATE TABLE statements"
            )
        ),
        Tool(
            name="Database Table Sample Data Lookup",
            func=data_table_samples_lookup_func,
            description=(
                "Useful to view sample data of a single table in the database. "
                "It displays column names, field types, and example data. "
                "Make sure the table exists in the database before using this command. "
                "Always use fully-qualified table names with this command."
                "Input should be a single fully-qualified table name. "
                "Output is a string representation of a JSON serialized object with "
                "field-mapped example data rows"
            )
        ),
        # Tool(
        #     name="Database Table Metadata Lookup",
        #     func=data_table_metadata_lookup_func,
        #     description=(
        #         "Useful to view the dbt configuration metadata of a single table. "
        #         "This includes table and column descriptions, as well as table lineage. "
        #         "Make sure the table exists in the database before using this command. "
        #         "Always use fully-qualified table names with this command."
        #         "Input should be a single fully-qualified table name. "
        #         "Output is a string with a json-serialized dictionary of the table metadata"
        #     )
        # ),
        Tool(
            name="Query ANSI-compliant SQL Database",
            func=query_sql_db,
            description=(
                "Before using this command, verify that all required table names and columns "
                "exist using other commands. Do not guess. Do not use this to query dataframes. "
                "Input should be a simple and valid ANSI-compliant SQL query using only the necessary columns. "
                "Use column aliases when using count or aggregation functions. "
                "The output will be a string with a json-serialized dictionary of the query "
                "results or a pandas dataframe with tabular data for further analysis and "
                "processing. If the query is incorrect, an error message will be returned. In "
                "case of an error, verify your assumptions and reformulate the ANSI-compliant "
                "SQL statement before trying again"
            )
        ),
        # Tool(
        #     name="General Knowledgebase Search",
        #     func=general_retrievalqa_func,
        #     description=(
        #         "Useful to gather additional context or background information needed for "
        #         "analysis which is not available by other commands. Use this before giving up."
        #         "Input should be a detailed fully formed question with as much context as possible."
        #         "Output is a json serialized dictionary with keys `answer` and `sources`"
        #     )
        # ),
        # Tool(
        #     name="Ask A Question",
        #     func=human_tool_func,
        #     return_direct=True,
        #     description=(
        #         "Useful to ask a human for help when you need more clarification or you are "
        #         "not sure how to complete the task. The input should be a question for the "
        #         "human. Only use this as a last resort when other commands are not sufficient"
        #     )
        # ),
        Tool(
            name="Proven SQL Statements Search",
            description=(
                "Useful to get example known-good SQL queries which are similar to a question. "
                "Input should be a detailed question with as much context as possible."
                "Output is a string with a list of similar SQL queries."
            ),
            func=get_similar_sql_queries_func
        ),
        Tool(
            name="Pandas Dataframe Tool (Read/Write)",
            description=(
                "This command initiates a Python REPL environment with the pandas library, "
                "which is used for manipulating dataframes. The dataframe is named `df`. "
                "Input must must be a complete and syntax-valid Python pandas expression "
                "that uses the `df` pandas DataFrame. The result of the expression will be "
                "automatically assigned back to the `df` dataframe."
                "The output will be a string representation of running df.head() after executing "
                "the command with the input. "
            ),
            func=python_func
        ),
        Tool(
            name="Pandas Dataframe Tool (Read Only)",
            description=(
                "Use this command to query the contents of a dataframe without modifying it. "
                "This command initiates a Python REPL environment with the pandas library, "
                "which is used for working with or reading the contents of dataframes. "
                "The dataframe is named `df`. Input must must be a complete and syntax-valid "
                "Python pandas expression that uses the `df` pandas DataFrame. "
                "The output will be a string representation of the output from running the "
                "command with the input. Do not use this command to modify the dataframe. "
            ),
            func=python_read_func
        )
    ]

    return tools
