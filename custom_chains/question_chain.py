"""Chain to generate a question to clarify the context of a request."""

from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from custom_prompts.chains import QUESTION_TEMPLATE
from custom_tools.llms import gpt4


class PydanticQuestionOutputResponse(BaseModel):
    """The response from the agent."""
    question: str = Field(
        description="The clarification question to ask the stakeholder"
    )
    dimensions: list[str] = Field(
        description="The dimensions needed to service the query"
    )
    metrics: list[str] = Field(
        description="The metrics needed to service the query"
    )
    granularity: str = Field(
        description="The granularity needed to service the query"
    )


QUESTION_PROMPT = PromptTemplate.from_template(QUESTION_TEMPLATE)

context_question_chain_output_parser = PydanticOutputParser(
    pydantic_object=PydanticQuestionOutputResponse)

context_question_chain = (
    QUESTION_PROMPT | gpt4 | context_question_chain_output_parser
)
