"""The chain for prescribing the output format of the visualization."""

from typing import List
from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from custom_prompts.chains import PRESCRIBE_OUTPUT_TEMPLATE
from custom_tools.llms import gpt35

PRESCRIBE_OUTPUT_FORMAT_PROMPT = PromptTemplate.from_template(PRESCRIBE_OUTPUT_TEMPLATE)

class PydanticPrescribeOutputResponse(BaseModel):
    """The response from the agent."""
    dimensions: List = Field(
        description="The dimensions needed to generate the visualization"
    )
    metrics: List = Field(
        description="The measures needed to generate the visualization"
    )
    granularity: str = Field(
        description="The granularity of the requested data needed for visualization"
    )

prescribe_output_parser = PydanticOutputParser(pydantic_object=PydanticPrescribeOutputResponse)

prescribe_chain = PRESCRIBE_OUTPUT_FORMAT_PROMPT | gpt35 | prescribe_output_parser
