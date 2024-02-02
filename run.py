"""Chatbot using a customized Langchain Structured Chat agent"""
from __future__ import annotations

import io
import os
from datetime import date

import json

from cryptography.fernet import Fernet

import pandas as pd
import numpy as np

from dotenv import load_dotenv, find_dotenv

import requests
import streamlit as st

from langchain.chains import LLMChain
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.schema import HumanMessage, AIMessage

from custom_tools.llms import gpt35
from custom_tools.pydantic_agent_output_parser import PydanticAgentOutputParser
from custom_tools import general
from custom_tools.general import (
    get_toolkit,
    get_similar_table_schemas_func,
    process_data,
    get_similar_sql_queries_func,
    format_sql_statement
)
from custom_chains.question_chain import context_question_chain
from custom_chains.condense_question_chain import condense_question_chain
from custom_chains.select_tool_chain import PydanticAgentResponse
from custom_agents.custom_structured_chat_agent_v4 import CustomStructuredChatAgentV4
from custom_prompts.agents import PREWORK_TEMPLATE_V4

# find the .env file and load it
# this sets OpenAI and other service API keys
load_dotenv(find_dotenv())

VERBOSE = os.getenv('VERBOSE') == 'true'


@st.cache_resource
def convert_df_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


TITLE = "Data Team Chatbot"
st.set_page_config(page_title=TITLE)
st.title(TITLE)

if 'session_id' not in st.session_state:
    key = Fernet.generate_key()
    f = Fernet(key)
    token = f.encrypt(b"session_id")

    # Convert the token to a string
    session_id = token.decode("utf-8")

    st.session_state['session_id'] = session_id

if 'current_date' not in st.session_state:
    today = date.today()
    # Month abbreviation, day and year
    st.session_state['current_date'] = today.strftime("%B %d, %Y")

# confrigure and load the agent tools
if 'toolkit' not in st.session_state:
    st.session_state['toolkit'] = get_toolkit()
    st.session_state['tool_names'] = [
        tool.name for tool in st.session_state['toolkit']]
    st.session_state['tool_names_list'] = ", ".join(
        "'" + item + "'" for item in st.session_state['tool_names'])

# The agent uses the agent chain, the allowed tools, and the output parser.
if 'agent' not in st.session_state:
    # The output parser is used to parse the agent's response and turn it into a command
    pydantic_agent_parser_extra_instructions = (
        "# Output format: \n"
        "```json\n"
        "{\n"
        "  \"reasoning\": $str_reasoning,\n"
        "  \"command\": $str_command,\n"
        "  \"command_input\": $str_command_input\n"
        "}\n"
        "```\n"
        "The '$str_reasoning' field must be a string and include the reasoning for the command. "
        "The '$str_command' field must be one of the following: "
        f"{st.session_state['tool_names_list']} or 'Final Answer'. "
        "The '$str_command_input' field must be a valid command input and not be empty.\n"
    )
    agent_parser = PydanticAgentOutputParser(
        pydantic_object=PydanticAgentResponse,
        extra_instructions=pydantic_agent_parser_extra_instructions
    )

    agent_prompt = CustomStructuredChatAgentV4.create_agent_prompt(
        tools=st.session_state['toolkit'],
        # format_instructions=agent_parser.get_format_instructions(),
        format_instructions=pydantic_agent_parser_extra_instructions,
        input_variables=[
            "query", "date", "dimensions", "metrics", "granularity",
            "agent_scratchpad", "agent_prework", "learned_context"
        ],
    )

    agent_chain = LLMChain(llm=gpt35, prompt=agent_prompt)

    st.session_state['agent'] = CustomStructuredChatAgentV4(
        llm_chain=agent_chain,
        output_parser=agent_parser,
        allowed_tools=st.session_state['tool_names'],
        handle_parsing_errors=True,
        max_iterations=20,
        max_iteration_time=1,
        early_stopping_method="generate",
    )

# The agent executor is used to orchestrate the agent and the tools
if 'agent_executor' not in st.session_state:
    st.session_state['agent_executor'] = AgentExecutor.from_agent_and_tools(
        agent=st.session_state['agent'],
        tools=st.session_state['toolkit'],
        handle_parsing_errors=False,
        verbose=VERBOSE
    )

if 'memory_storage' not in st.session_state:
    st.session_state['memory_storage'] = StreamlitChatMessageHistory()
    st.session_state['history_memory'] = ConversationBufferMemory(
        return_messages=True,
        output_key="answer",
        input_key="question",
        chat_memory=st.session_state['memory_storage'])


def get_chat_history(memory_storage: StreamlitChatMessageHistory):
    """Get the chat memory storage from the Streamlit session."""
    buffer = "\n"
    for msg in st.session_state['memory_storage'].messages:
        if isinstance(msg, HumanMessage):
            buffer += "Human: " + msg.content
        elif isinstance(msg, AIMessage):
            content = json.loads(msg.content)["response"]
            buffer += "Assistant: " + content
        buffer += "\n"
    return buffer


if 'learned_context' not in st.session_state:
    st.session_state['learned_context'] = general.learned_context

if 'sql_statement' not in st.session_state:
    st.session_state['sql_statement'] = general.sql_statement

if 'df' not in st.session_state:
    st.session_state['df'] = general.df

if 'assumptions' not in st.session_state:
    st.session_state['assumptions'] = ""

if 'csv' not in st.session_state:
    st.session_state['csv'] = None

if 'image' not in st.session_state:
    st.session_state['image'] = None

if 'graph_code' not in st.session_state:
    st.session_state['graph_code'] = None

if 'stage' not in st.session_state:
    st.session_state['stage'] = 0

if st.button("Reset"):
    st.session_state = {}
    st.rerun()

# The Streamlit chat app container
chat_container = st.container()
with chat_container:
    human_messages = st.session_state['memory_storage'].messages[::2]
    ai_messages = st.session_state['memory_storage'].messages[1::2]
    message_list = zip(ai_messages, human_messages)
    for i, (ai_msg, human_msg) in enumerate(message_list):
        st.chat_message("user").write(human_msg.content)
        ai = json.loads(ai_msg.content)
        with st.chat_message("assistant"):
            with st.expander("SQL Statement"):
                st.markdown(ai["sql_statement"])
            with st.expander("Learned Context"):
                st.markdown(ai["learned_context"])
            if ai["image"]:
                st.image(ai["image"])
            if ai["csv"]:
                with st.expander("Dataframe"):
                    # download csv from url
                    csv_url = ai["csv"]
                    response = requests.get(csv_url, timeout=5)
                    csv_data = response.content.decode('utf-8')
                    df = pd.read_csv(io.StringIO(csv_data))
                    st.dataframe(df)
                    st.download_button(
                        key="download_button_" + str(i),
                        label="Download CSV",
                        data=convert_df_to_csv(df),
                        file_name="data.csv",
                        mime="text/csv"
                    )
            st.write(ai["response"])
    user_container = st.container()
    assistant_container = st.container()

# render the chat input
st.session_state['current_input'] = st.chat_input(
    placeholder="Ask me anything", key="input")

# run the orchestration agent when the user submits a question
if user_input := st.session_state.get('current_input'):
    user_container.empty()
    assistant_container.empty()
    st.session_state['current_input'] = None
    with user_container:
        with st.chat_message("user"):
            st.markdown(user_input)
    with assistant_container:
        with st.chat_message("assistant"):
            with st.spinner('Processing...'):
                response = condense_question_chain.invoke(
                    {
                        "answer": user_input,
                        "date": st.session_state['current_date'],
                        "chat_history": get_chat_history(
                            st.session_state['memory_storage'])
                    }
                )
                st.session_state['user_query'] = response.question
                st.session_state['assumptions'] = response.assumptions

                if VERBOSE:
                    print("Condensed question:", response.question)
                    print("\n")
                    print("Assumptions:", response.assumptions)

                response = context_question_chain.invoke(
                    {
                        "question": st.session_state['user_query'],
                        "date": st.session_state['current_date'],
                    }
                )

                if "no more questions" not in response.question.lower():
                    response = response.question
                else:
                    if st.session_state['assumptions']:
                        with st.expander("Assumptions"):
                            st.write(st.session_state['assumptions'])

                    if st.session_state['stage'] < 2:
                        st.session_state['dimensions'] = str(
                            response.dimensions)
                        st.session_state['metrics'] = str(response.metrics)
                        st.session_state['granularity'] = response.granularity
                        similar_table_schemas = get_similar_table_schemas_func(
                            {
                                "query": st.session_state['user_query'],
                                "requested_dimensions": st.session_state['dimensions'],
                                "requested_metrics": st.session_state['metrics'],
                                "requested_granularity": st.session_state['granularity'],
                            }
                        )
                        agent_prework = PREWORK_TEMPLATE_V4.format(
                            reasoning=(
                                "To find the data I need I should search for table schemas "
                                "similar to the query."),
                            command="Database Table Schema Search",
                            command_input=st.session_state['user_query'],
                            command_output=similar_table_schemas
                        )
                        similar_sql_statements = get_similar_sql_queries_func(
                            query=st.session_state['user_query']
                        )
                        agent_prework += PREWORK_TEMPLATE_V4.format(
                            reasoning=(
                                "I should also search for known-good SQL queries for questions "
                                "similar to the query."),
                            command="Proven SQL Statements Search",
                            command_input=st.session_state['user_query'],
                            command_output=similar_sql_statements
                        )
                    else:
                        agent_prework = ""

                    agent_input = {
                        "query": st.session_state['user_query'],
                        "date": st.session_state['current_date'],
                        "dimensions": st.session_state['dimensions'],
                        "metrics": st.session_state['metrics'],
                        "granularity": st.session_state['granularity'],
                        "agent_prework": agent_prework,
                        "learned_context": st.session_state['learned_context']
                    }
                    response = st.session_state['agent_executor'](agent_input)[
                        'output']

                    if general.learned_context != "None":
                        st.session_state['learned_context'] = general.learned_context
                        with st.expander("Learned Context"):
                            st.markdown(
                                st.session_state['learned_context'].replace("$", r"\$"))

                    if general.sql_statement:
                        st.session_state['sql_statement'] = format_sql_statement(
                            general.sql_statement)
                        with st.expander("SQL Statement"):
                            st.markdown(st.session_state['sql_statement'])

                    st.session_state['stage'] += 1

                    if isinstance(general.df, pd.DataFrame):
                        if general.df.size > 2:
                            try:
                                general.df = general.df.fillna(0)
                            except:
                                pass
                            for col in general.df.select_dtypes('Int64').columns:
                                general.df[col] = general.df[col].astype(
                                    np.int64)
                            st.session_state['df'] = general.df
                            st.session_state['graph_code'], st.session_state['image'], st.session_state['csv'] = process_data(
                                {
                                    "question": st.session_state['user_query'],
                                    "dimensions": st.session_state['dimensions'],
                                    "metrics": st.session_state['metrics'],
                                    "granularity": st.session_state['granularity'],
                                    "df": st.session_state['df'],
                                    "session_id": st.session_state['session_id'],
                                    "stage": st.session_state['stage']
                                }
                            )
                            if st.session_state['image']:
                                st.image(st.session_state['image'])
                            if st.session_state['graph_code'] and VERBOSE:
                                with st.expander("Visualization code"):
                                    st.code(st.session_state['graph_code'])
                            with st.expander("Dataframe"):
                                st.dataframe(general.df)
                                st.download_button(
                                    key="download_button",
                                    label="Download CSV",
                                    data=convert_df_to_csv(
                                        st.session_state['df']),
                                    file_name="data.csv",
                                    mime="text/csv"
                                )
                        elif general.df.size == 2:
                            st.table(general.df)
                        elif general.df.size == 1:
                            response = general.df.iat[0, 0]

                # Display the response from the question chain
                if isinstance(response, str):
                    response = response.replace("$", r"\$")
                else:
                    response = str(response)
                st.markdown(response)
                if general.df.size:
                    st.session_state['stage'] = 2
                    _, _, col1, col2, _ = st.columns(5)
                    with col1:
                        st.button("👍")
                    with col2:
                        st.button("👎")

    st.session_state['history_memory'].save_context(
        {"question": user_input}, {"answer": json.dumps({
            "response": response,
            "sql_statement": st.session_state['sql_statement'],
            "learned_context": st.session_state['learned_context'],
            "csv": st.session_state['csv'],
            "image": st.session_state['image']
        })}
    )
