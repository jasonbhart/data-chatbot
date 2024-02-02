"""Chain for imagining a table schema from a natural language question."""

from operator import itemgetter

from typing import List, Union

from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableMap

from custom_prompts.chains import SCHEMA_SEARCH_TEMPLATE
from custom_tools.llms import gpt35, gpt4

SCHEMA_SEARCH_TEMPLATE_PROMPT = PromptTemplate.from_template(
    SCHEMA_SEARCH_TEMPLATE)


class PydanticImagineTablesOutputResponse(BaseModel):
    """The response from the agent."""
    table_schemas_converted_to_create_table_statements: Union[str, List[str]] = Field(
        description="The ideal table schema to service the query represented by CREATE TABLE statements"
    )


schema_search_output_parser = PydanticOutputParser(
    pydantic_object=PydanticImagineTablesOutputResponse)

loaded_schema_search_variables = RunnableMap(
    {
        "query": itemgetter("query"),
    }
)

schema_search_chain = (
    loaded_schema_search_variables |
    SCHEMA_SEARCH_TEMPLATE_PROMPT |
    gpt35 |
    schema_search_output_parser
)
