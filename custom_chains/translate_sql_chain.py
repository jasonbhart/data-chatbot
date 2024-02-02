"""The chain for translating a SQL statement to a native dialect."""

from typing import List
from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from custom_prompts.chains import TRANSLATE_SQL_TEMPLATE
from custom_tools.llms import gpt35, gpt4

TRANSLATE_SQL_TEMPLATE_PROMPT = PromptTemplate.from_template(
    TRANSLATE_SQL_TEMPLATE)


class PydanticTranslateSQLResponse(BaseModel):
    """The response from the agent."""
    reasoning: str = Field(
        description="The description of the problem and reasoning behind the translation"
    )
    sql_statement: str = Field(
        description="The fixed SQL statement"
    )


translate_sql_output_parser = PydanticOutputParser(
    pydantic_object=PydanticTranslateSQLResponse)

translate_sql_chain = TRANSLATE_SQL_TEMPLATE_PROMPT | gpt4 | translate_sql_output_parser
