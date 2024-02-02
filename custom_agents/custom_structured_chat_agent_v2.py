from typing import Any, Dict, List, Optional, Sequence, Tuple

import os
import re

from dotenv import load_dotenv, find_dotenv

import tiktoken

from langchain.chains import LLMChain
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.agents import StructuredChatAgent
from langchain.schema import AgentAction, BasePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.tools import BaseTool

VERBOSE = True

# find the .env file and load it
# this sets OpenAI and other service API keys
load_dotenv(find_dotenv())

# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def count_num_tokens(string: str) -> int:
    """Returns the number of tokens"""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def edit_scratchpad_for_relevance(scratchpad: str, query: str) -> str:
    """Use LLM to remove parts of the scratchpad that are not relevant to the query"""

    edit_llm_template = """
Given a query and a log of work done to service that query, remove any irrelevant
parts of the log, and return the results. Keep any parts of the log that help
disprove assumptions made during the process of answering the query to avoid
repeating mistakes. For the parts of the log which are removed, add a comment
in their place concisely summarizing the work and explaining why it was removed.
Make sure to preserve any meaningful structure and sequencing present in the log.

Query: {query}

Log:
{log}
    """

    edit_llm = LLMChain(
        llm=tool_llm, prompt=PromptTemplate.from_template(edit_llm_template))
    scratchpad = edit_llm.run({"log": scratchpad, "query": query})
    return scratchpad


tool_llm = PromptLayerChatOpenAI(
    model="gpt-3.5-turbo-16k",
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    pl_tags=["langchain", "datateam-chatbot", "normal-agent_v2", "tool"],
    verbose=VERBOSE
)

PREFIX = """
# Objective:
As a Springboard data analyst, efficiently address human queries by:
1. Identifying the query's intent and expected output.
2. From the query, research and find the appropriate data tables needed to service the query.
3. Once you have the list of table names you need, get the tables' schemas, metadata, and example data.
4. Only then, given everything you know now, formulate your SQL query and run it against the database.
5. Use the available tools to extract, process, and check your data.
6. Transform the data into the expected output format, if it isn't already.
7. Provide the output to the human in the expected format.

Avoid unnecessary steps. Heed criticism when thinking about what to do next. Don't repeat work that has already been done.

## Commands Available:"""

SUFFIX = """
## Example:

### Query: input question to answer
### Thought: consider previous and subsequent steps
### Criticism: what else should be considered?
### Command:
```
$JSON_OBJECT
```
### Result:
*command result*

... *(repeat Thought/Criticism/Command/Result N times)*

### Thought: I know how to respond to the query
### Criticism: I have exhausted all considerations
### Command:
```
{{{{
  "command": "Final Answer",
  "command_input": "Final response to human"
}}}}
```

Remember- You must ALWAYS include one $JSON_OBJECT for every Command section in your response!

"""

HUMAN_MESSAGE_TEMPLATE = "Begin!\n\n### Query: {input}{agent_scratchpad}"


class CustomStructuredChatAgentV2(StructuredChatAgent):
    """Structured Chat Agent."""

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "### Result:\n"

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "\n### Thought:"

    @property
    def _stop(self) -> List[str]:
        return ["### Result:"]

    def get_full_inputs(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps."""
        thoughts = self._construct_scratchpad(
            intermediate_steps, kwargs["input"])
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        full_inputs = {**kwargs, **new_inputs}
        return full_inputs

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]], input_kwarg: str = ""
    ) -> str:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n{self.observation_prefix}{observation}"
        agent_scratchpad = str(thoughts)
        if count_num_tokens(agent_scratchpad) > 8000:
            # text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000)
            # splits = text_splitter.split_text(agent_scratchpad)
            # if isinstance(splits, list):
            #     agent_scratchpad = splits[0]
            agent_scratchpad = edit_scratchpad_for_relevance(
                scratchpad=agent_scratchpad,
                query=input_kwarg
            )
        return agent_scratchpad

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        human_message_template: str = HUMAN_MESSAGE_TEMPLATE,
        format_instructions: str = None,
        input_variables: Optional[List[str]] = None,
        memory_prompts: Optional[List[BasePromptTemplate]] = None,
    ) -> BasePromptTemplate:
        tool_strings = []
        for tool in tools:
            args_schema = re.sub(
                "}", "}}}}", re.sub("{", "{{{{", str(tool.args)))
            tool_strings.append(
                f"{tool.name}: {tool.description}, args: {args_schema}")
        formatted_tools = "\n".join(tool_strings)
        tool_names = ", ".join([tool.name for tool in tools])
        tool_names += ", Final Answer"
        format_instructions = re.sub(
            "}", "}}}}", re.sub("{", "{{{{", format_instructions))
        format_instructions += f"The \"command\" field must be one of the following: {tool_names}"
        template = "\n\n".join(
            [prefix, formatted_tools, format_instructions, suffix])
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]
        _memory_prompts = memory_prompts or []
        messages = [
            SystemMessagePromptTemplate.from_template(template),
            *_memory_prompts,
            HumanMessagePromptTemplate.from_template(human_message_template),
        ]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)
